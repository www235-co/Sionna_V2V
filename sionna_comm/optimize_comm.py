import os
import gc
import torch
import pandas
import numpy as np
import pulp as lp
from sionna_comm.calculate_throughput import CalculateThroughput

def optimize_comm(scenario_name, frame_id, current_frame_stage1, hypes, mode=3):
    # mode 1: time-frequency 
    # mode 2: only time
    # mode 3: only frequency


    tp_save_path = os.path.join("comm", "saved_tp", f"{scenario_name}_mode{mode}.npy")
    os.makedirs(os.path.dirname(tp_save_path), exist_ok=True)

    config_path = os.path.join("sionna_comm", "comm_config.yaml")
    distance_path = os.path.join(hypes['validate_dir'], f"{scenario_name}")
    tp_calculator = CalculateThroughput(config_path, distance_path)


    if os.path.exists(tp_save_path):
        print("youle")
        tp_merged = np.load(tp_save_path)  
    
    else:
        channel_types = ['los', 'nlos', 'nlosv']
        tp_dict = {}
        for channel_type in channel_types:  
            tp_slot_subcarrier = tp_calculator.calculate_throughput_slot_subcarrier(channel_type)
            
            if 1 == mode:
                tp = tp_calculator.calculate_throughput_slot_subchannel(tp_slot_subcarrier) # [num_frame, slot_per_frame, num_subchannel, num_veh , num_veh]
                tp_dict[channel_type] = tp
            elif 2 == mode:
                tp = tp_calculator.calculate_throughput_slot_all_subchannel(tp_slot_subcarrier) # [num_frame, slot_per_frame, num_veh , num_veh]
                tp_dict[channel_type] = tp
            elif 3 == mode:
                tp = tp_calculator.calculate_throughput_subchannel_all_slot(tp_slot_subcarrier) # [num_frame, slot_per_frame, num_subchannel, num_veh , num_veh]
                tp_dict[channel_type] = tp
            else:
                raise ValueError("mode is invalid")
        tp_los = tp_dict['los']
        tp_nlos = tp_dict['nlos']
        tp_nlosv = tp_dict['nlosv']

        tp_merged = tp_calculator.merge_tp(tp_los, tp_nlos, tp_nlosv, mode=mode)
        np.save(tp_save_path, tp_merged)
    
    if mode == 1 or mode == 2:

        ego_indices = list(current_frame_stage1.keys())
        N = len(ego_indices)
        communication_masks = {}  

        score_arrays = {} 
        psm_arrays = {}   


        for current_ego_idx in range(N):
            stage1_res = current_frame_stage1[current_ego_idx]

            score_map_np = tensor_list_to_numpy(stage1_res['score_maps'], is_spatial=True)
            psm_np = tensor_list_to_numpy(stage1_res['communication_maps'], is_spatial=True)

            L, C, H, W = psm_np.shape 

            score_arrays[current_ego_idx] = score_map_np
            psm_arrays[current_ego_idx] = psm_np

        tau_min, tau_max = 0.0, 1.0  
        best_total_utility = -1.0  
        best_comm_matrix = np.zeros((N, N), dtype=int)
        best_masks = {i: np.zeros_like(psm_arrays[i]) for i in range(N)}
        best_total_comm_volume = 0
        best_comm_volume_matrix = np.zeros((N, N), dtype=int)

        for iter_idx in range(tp_calculator.iter_times):
            tau = (tau_min + tau_max) / 2  
            comm_matrix = np.zeros((N, N), dtype=int) 
            current_masks = {} 
            current_throughput_matrix = np.zeros((N, N), dtype=np.float32)
            current_total_comm_volume = 0  
            current_comm_volume_matrix = np.zeros((N, N), dtype=int)
            current_total_utility = 0.0 

            communication_tasks = []

            for current_ego_id in range(N):
                score = score_arrays[current_ego_id]
                L, C, H, W = psm_arrays[current_ego_id].shape

                current_best_mask = np.zeros((L, 1, H, W), dtype=np.float32)
                current_best_mask[current_ego_id, ...] = 1.0  
                sum_selected_utility = 0.0  
                partner_candidates = []    

                for partner_pos in range(N):
                    if current_ego_id == partner_pos:
                        continue  

                    partner_score = score[partner_pos, ...].squeeze()  # (H,W)
                    partner_mask_2d = (partner_score > tau).astype(np.float32)
                    n_grids = int(np.sum(partner_mask_2d)) 
                    n_bits = n_grids * tp_calculator.bits_per_grid  

                    utility = cal_utility(partner_score, partner_mask_2d)
                    
                    partner_candidates.append( ( -utility, partner_pos, partner_mask_2d, n_bits ) )

                selected_partners = []
                if len(partner_candidates) > 0:
                    partner_candidates.sort()
                    take_num = min(tp_calculator.num_partner, len(partner_candidates)) 
                    selected_partners = partner_candidates[:take_num]

                for i, partner_pos, partner_mask_2d, n_bits in selected_partners:
                    current_best_mask[partner_pos, 0, ...] = partner_mask_2d
                    sum_selected_utility -= i
                    comm_matrix[current_ego_id][partner_pos] = 1
                    current_comm_volume_matrix[current_ego_id][partner_pos] = n_bits
                    current_total_comm_volume += n_bits
                    communication_tasks.append( (current_ego_id, partner_pos, n_bits) )

                current_masks[current_ego_id] = current_best_mask
                current_total_utility += sum_selected_utility

            if mode == 1:
                available_flag = False
                
                active_tasks = [task for task in communication_tasks if task[1] != -1]
                M = len(active_tasks)  
                SCH = tp_calculator.num_subchannel 
                G_SET = int(tp_calculator.effective_slot_per_frame/tp_calculator.num_slot_per_group)

                if M == 0:
                    available_flag = True
                else:
                    I = range(M)  
                    S = range(SCH)
                    G = range(G_SET)
                    Vehicles = range(N)

                    D = [task[2] for task in active_tasks]  
                    r = np.zeros((M, G_SET, SCH), dtype=np.float32)
                    for i in I:
                        receiver, sender, _ = active_tasks[i]
                        for g in G:
                            for s in S:
                                r[i, g, s] = tp_merged[frame_id, g, s, receiver, sender]

                    prob = lp.LpProblem("Vehicle_Communication_Scheduling", lp.LpMinimize)

                    prob += 0, "No_Optimize_Objective"

                    x = lp.LpVariable.dicts(
                        name="x",
                        indices=(I, G, S),
                        cat=lp.LpBinary, 
                    )

                    is_tx = lp.LpVariable.dicts("is_tx", (Vehicles, G), cat=lp.LpBinary)
                    is_rx = lp.LpVariable.dicts("is_rx", (Vehicles, G), cat=lp.LpBinary)

                    for g in G:
                        for v in Vehicles:
                            prob += is_tx[v][g] + is_rx[v][g] <= 1, f"HalfDuplex_Veh{v}_Group{g}"
                    
                    for i in I:
                        receiver_id, sender_id, _ = active_tasks[i]

                        for g in G:
                            for s in S:

                                prob += x[i][g][s] <= is_tx[sender_id][g], f"Link_Tx_Task{i}_Veh{sender_id}_G{g}_S{s}"
                                
                                prob += x[i][g][s] <= is_rx[receiver_id][g], f"Link_Rx_Task{i}_Veh{receiver_id}_G{g}_S{s}"

                    for i in I:
                        prob += (
                            lp.lpSum(x[i][g][s] * r[i, g, s] for g in G for s in S) >= D[i],
                            f"Demand_Constraint_Task_{i}"
                        )

                    for g in G:
                        for s in S:
                            prob += (
                                lp.lpSum(x[i][g][s] for i in I) <= 1,
                                f"Channel_Capacity_Constraint_resource_{g}_{s}"
                            )
                    
                    prob.solve(lp.COIN_CMD(msg=1, timeLimit=30))

                    solve_status = lp.LpStatus[prob.status]

                    import pandas as pd
                    if solve_status == "Optimal":
                        allocation_data = []
                        
                        for g in G:
                            for s in S:
                                entry = {
                                    "Time_Group_G": g,
                                    "Subchannel_S": s,
                                    "Allocated_Task": "None",
                                    "Actual_Throughput": 0,
                                    "Sender": "N/A",
                                    "Receiver": "N/A",
                                    "Task_Index": -1
                                }
                                
                                for i in I:
                                    receiver_idx, sender_idx, initial_demand = active_tasks[i]
                                    entry[f"T{i}_Rate_{sender_idx}->{receiver_idx}"] = r[i, g, s]
                                    entry[f"T{i}_Initial_Demand"] = initial_demand
                                    
                                    if lp.value(x[i][g][s]) == 1:
                                        entry["Allocated_Task"] = f"Task_{i}"
                                        entry["Actual_Throughput"] = r[i, g, s]
                                        entry["Sender"] = sender_idx
                                        entry["Receiver"] = receiver_idx
                                        entry["Task_Index"] = i
                                
                                allocation_data.append(entry)

                        df_alloc = pd.DataFrame(allocation_data)
                        
                        output_dir = f"./sionna_comm/allocation_logs/Time_Freq/{scenario_name}/frame_{frame_id}/"
                        os.makedirs(output_dir, exist_ok=True)
                        file_name = f"iter_{iter_idx}_tau_{tau:.4f}_Optimal.xlsx"
                        save_path = os.path.join(output_dir, file_name)
                        
                        df_alloc.to_excel(save_path, index=False)

                        available_flag = True  
                    elif solve_status == "Infeasible":
                        print(f"no feasible solution")

                    del prob, x, r
                    gc.collect()
            
            if mode == 2:
                available_flag = False
                total_time = 0
                
                active_tasks = [task for task in communication_tasks if task[1] != -1]
                M = len(active_tasks)  
                if M == 0:
                    available_flag = True
                else:
                    I = range(M)  
                    T_set = range(1, tp_calculator.effective_slot_per_frame + 1)  
                    D = [task[2] for task in active_tasks] 
                    r = np.zeros((M, tp_calculator.effective_slot_per_frame), dtype=np.float32)
                    for i in I:
                        sender, receiver, _ = active_tasks[i]
                        for t in T_set:
                            r[i, t-1] = tp_merged[frame_id, t-1, sender, receiver]

                    prob = lp.LpProblem("Vehicle_Communication_Scheduling", lp.LpMinimize)

                    prob += 0, "No_Optimize_Objective"

                    x = lp.LpVariable.dicts(
                        name="x",
                        indices=(I, T_set),
                        cat=lp.LpBinary,  
                    )

                    for i in I:
                        prob += (
                            lp.lpSum(x[i][t] * r[i, t-1] for t in T_set) >= D[i],
                            f"Demand_Constraint_Task_{i}"
                        )

                    for t in T_set:
                        prob += (
                            lp.lpSum(x[i][t] for i in I) <= 1,
                            f"Channel_Capacity_Constraint_Slot_{t}"
                        )

                    prob.solve(lp.COIN_CMD(msg=1, timeLimit=15))

                    solve_status = lp.LpStatus[prob.status]
                    if solve_status == "Optimal":
                        used_slots = []
                        for i in I:
                            for t in T_set:
                                if lp.value(x[i][t]) == 1:
                                    used_slots.append(t)
                        available_flag = True  
                    elif solve_status == "Infeasible":
                        print(f"no feasible solution")

                    del prob, x, r
                    gc.collect()

            if available_flag:
                if current_total_utility > best_total_utility:
                    best_total_utility = current_total_utility
                    best_comm_matrix = comm_matrix.copy()
                    best_masks = current_masks.copy()
                    best_total_comm_volume = current_total_comm_volume
                    best_comm_volume_matrix = current_comm_volume_matrix.copy()
                tau_max = tau
            else:
                tau_min = tau


        for ego_id in ego_indices:
            communication_masks[ego_id] = torch.from_numpy(best_masks[ego_id])

        for ego_idx in ego_indices:
            mask_tensor = communication_masks[ego_idx]
            assert mask_tensor.ndim == 4
            L_mask, C_mask, H_mask, W_mask = mask_tensor.shape
            L_orig, C_orig, H_orig, W_orig = psm_arrays[ego_idx].shape
            assert (L_mask, C_mask, H_mask, W_mask) == (L_orig, C_orig, H_orig, W_orig)

        return (
            best_comm_matrix,
            communication_masks,
            best_total_comm_volume,
            best_comm_volume_matrix,
            tau
        )

    if 3 == mode:
        ego_indices = list(current_frame_stage1.keys())
        N = len(ego_indices)
        communication_masks = {}  

        score_arrays = {}  
        psm_arrays = {}    

        for current_ego_idx in range(N):
            stage1_res = current_frame_stage1[current_ego_idx]
            score_map_np = tensor_list_to_numpy(stage1_res['score_maps'], is_spatial=True)
            psm_np = tensor_list_to_numpy(stage1_res['communication_maps'], is_spatial=True)
            L, C, H, W = psm_np.shape
            score_arrays[current_ego_idx] = score_map_np
            psm_arrays[current_ego_idx] = psm_np

        tau_min, tau_max = 0.0, 1.0
        best_total_utility = -1.0
        best_transmitter = -1  
        best_comm_matrix = np.zeros((N, N), dtype=int)
        best_masks = {i: np.zeros_like(psm_arrays[i]) for i in range(N)}
        best_total_comm_volume = 0
        best_comm_volume_matrix = np.zeros((N, N), dtype=int)

        for iter_idx in range(tp_calculator.iter_times):
            tau = (tau_min + tau_max) / 2
            
            current_total_utility = 0.0
            current_total_comm_volume = 0
            current_comm_volume_matrix = np.zeros((N, N), dtype=int) 
            communication_tasks = []

            transmitter_candidates = []
            
            for candidate_transmitter in range(N):

                total_utility_as_transmitter = 0.0
                transmitter_tasks = []  
                
                for receiver in range(N):
                    if candidate_transmitter == receiver:
                        continue  
                    
                    receiver_score = score_arrays[receiver]
                    sender_score_in_receiver_view = receiver_score[candidate_transmitter, ...].squeeze()  # (H,W)
                    
                    sender_mask_2d = (sender_score_in_receiver_view > tau).astype(np.float32)
                    n_grids = int(np.sum(sender_mask_2d))
                    n_bits = n_grids * tp_calculator.bits_per_grid
                    
                    utility = cal_utility(sender_score_in_receiver_view, sender_mask_2d)
                    total_utility_as_transmitter += utility
                    
                    transmitter_tasks.append({
                        'receiver': receiver,
                        'sender': candidate_transmitter,
                        'n_bits': n_bits,
                        'mask_2d': sender_mask_2d,
                        'utility': utility
                    })
                
                transmitter_candidates.append({
                    'transmitter': candidate_transmitter,
                    'total_utility': total_utility_as_transmitter,
                    'tasks': transmitter_tasks
                })
            
            if transmitter_candidates:
                best_candidate = max(transmitter_candidates, key=lambda x: x['total_utility'])
                selected_transmitter = best_candidate['transmitter']
                current_total_utility = best_candidate['total_utility']
                
                comm_matrix = np.zeros((N, N), dtype=int)
                for task in best_candidate['tasks']:
                    receiver = task['receiver']
                    comm_matrix[selected_transmitter][receiver] = 1
                
                current_masks = {}
                
                for ego_id in range(N):
                    L, C, H, W = psm_arrays[ego_id].shape
                    base_mask = np.zeros((L, 1, H, W), dtype=np.float32)
                    base_mask[ego_id, ...] = 1.0  
                    
                    if ego_id != selected_transmitter:
                        for task in best_candidate['tasks']:
                            if task['receiver'] == ego_id:
                                sender_mask_2d = task['mask_2d']
                                base_mask[selected_transmitter, 0, ...] = sender_mask_2d
                                current_comm_volume_matrix[selected_transmitter][ego_id] = task['n_bits']
                                current_total_comm_volume += task['n_bits']
                                break
                    
                    current_masks[ego_id] = base_mask
                

                for task in best_candidate['tasks']:
                    communication_tasks.append((task['receiver'], task['sender'], task['n_bits']))

            else:
                selected_transmitter = -1
                comm_matrix = np.zeros((N, N), dtype=int)
                current_masks = {i: np.zeros_like(psm_arrays[i]) for i in range(N)}

            available_flag = False
            active_tasks = communication_tasks 
            M = len(active_tasks)
            SCH = tp_calculator.num_subchannel

            if M == 0:
                available_flag = True
            else:
                I = range(M)
                S = range(SCH)

                D = [task[2] for task in active_tasks]
                r = np.zeros((M, SCH), dtype=np.float32)
                for i in I:
                    receiver, sender, _ = active_tasks[i]
                    for s in S:
                        r[i, s] = tp_merged[frame_id, s, receiver, sender]

                prob = lp.LpProblem("Vehicle_Communication_Scheduling_OneToMany", lp.LpMinimize)
                prob += 0, "No_Optimize_Objective"

                x = lp.LpVariable.dicts(
                    name="x",
                    indices=(I, S),
                    cat=lp.LpBinary,
                )

                for i in I:
                    prob += (
                        lp.lpSum(x[i][s] * r[i, s] for s in S) >= D[i],
                        f"Demand_Constraint_Task_{i}"
                    )

                for s in S:
                    prob += (
                        lp.lpSum(x[i][s] for i in I) <= 1,
                        f"Channel_Capacity_Constraint_subchannel_{s}"
                    )

                prob.solve(lp.COIN_CMD(msg=1, timeLimit=5))

                solve_status = lp.LpStatus[prob.status]

                if solve_status == "Optimal":
                    allocation_data = []
                    for s in S:
                        entry = {
                            "Subchannel": s,
                            "Allocated_Task": "None",
                            "Actual_Throughput": 0,
                            "Sender": "N/A",
                            "Receiver": "N/A",
                            "Task_ID": -1
                        }
                        
                        for i in I:
                            receiver, sender, demand = active_tasks[i]
                            entry[f"T{i}_Rate_{sender}->{receiver}"] = r[i, s]
                            entry[f"T{i}_Initial_Demand"] = demand
                            
                            if lp.value(x[i][s]) == 1:
                                entry["Allocated_Task"] = f"Task_{i}"
                                entry["Actual_Throughput"] = r[i, s]
                                entry["Sender"] = sender
                                entry["Receiver"] = receiver
                                entry["Task_ID"] = i
                        
                        allocation_data.append(entry)
                    
                    df = pandas.DataFrame(allocation_data)
                    output_dir = f"./sionna_comm/allocation_logs_one_to_many/{scenario_name}/frame_{frame_id}/"
                    os.makedirs(output_dir, exist_ok=True)
                    file_name = f"iter_{iter_idx}_tau_{tau:.4f}_sender_{selected_transmitter}.xlsx"
                    df.to_excel(os.path.join(output_dir, file_name), index=False)
                    
                    available_flag = True
                elif solve_status == "Infeasible":
                    print(f"no feasible solution")
                    available_flag = False


                del prob, x, r
                gc.collect()

            if available_flag:
                if current_total_utility > best_total_utility:
                    best_total_utility = current_total_utility
                    best_transmitter = selected_transmitter
                    best_comm_matrix = comm_matrix.copy()
                    best_masks = current_masks.copy()
                    best_total_comm_volume = current_total_comm_volume
                    best_comm_volume_matrix = current_comm_volume_matrix.copy()
                    for ego_id in range(N):
                        mask = best_masks[ego_id]
                        non_zero_count = np.sum(mask > 0)
                tau_max = tau 
            else:
                tau_min = tau 

        final_tau = (tau_min + tau_max) / 2

        for ego_id in ego_indices:
            communication_masks[ego_id] = torch.from_numpy(best_masks[ego_id])

        for ego_idx in ego_indices:
            mask_tensor = communication_masks[ego_idx]
            assert mask_tensor.ndim == 4

        return (
            best_comm_matrix,
            communication_masks,  
            best_total_comm_volume,
            best_comm_volume_matrix,
            final_tau
        )

def tensor_list_to_numpy(nested_data, is_spatial=False):
    if isinstance(nested_data, (list, tuple)):
        processed = [tensor_list_to_numpy(item, is_spatial) for item in nested_data]
        if is_spatial and len(processed) > 0 and isinstance(processed[0], np.ndarray):
            return np.concatenate(processed, axis=0) 
        return processed
    elif isinstance(nested_data, torch.Tensor):
        numpy_array = nested_data.cpu().numpy()
        if is_spatial:
            if numpy_array.ndim == 3:
                numpy_array = numpy_array[np.newaxis, ...] 
            assert numpy_array.ndim == 4
        return numpy_array
    else:
        arr = np.array(nested_data)
        if is_spatial:
            if arr.ndim == 3:
                arr = arr[np.newaxis, ...] 
            assert arr.ndim == 4
        return arr
    

def cal_utility(partner_score, partner_mask):
    assert partner_score.ndim == 2 and partner_mask.ndim == 2 
    return np.sum(partner_score * partner_mask )



