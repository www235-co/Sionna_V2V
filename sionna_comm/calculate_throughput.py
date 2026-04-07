import re
import yaml
import random
import sionna
import numpy as np
import tensorflow as tf
from sionna_comm.CDL import CDL
from sionna.phy.channel.tr38901 import PanelArray
from sionna.phy.utils import db_to_lin, lin_to_db
from sionna.phy.channel.utils import subcarrier_frequencies, cir_to_ofdm_channel
from sionna_comm.build_distance_tensor import build_vehicle_distance_tensor
from sionna_comm.large_scale_fading import LargeScaleFading
from sionna_comm.utils import reconstruct

ALPHA = 0.6
NUM_SLOT_PER_GROUP = 10

loader = yaml.SafeLoader
loader.add_implicit_resolver(u'tag:yaml.org,2002:float',
                            re.compile(u'''^(?:
                             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                            |[-+]?\\.(?:inf|Inf|INF)|\\.(?:nan|NaN|NAN))$''', re.X),
                            list(u'-+0123456789.'))


class CalculateThroughput:
    def __init__(self, config_path, distance_path):
        self.distance_path = distance_path
        self.config_path = config_path
        self._parse_config()
        self._set_seed()
        self.distance, self.nlos_mask, self.frame_map, self.vehicle_map = build_vehicle_distance_tensor(self.distance_path)
        self.distance = tf.expand_dims(self.distance,axis=1)
        self.distance = tf.repeat(self.distance,repeats=self.slot_per_frame,axis=1) 

        self.num_frame, _, self.num_vehicle, _ = self.distance.shape 
        self.num_subcarrier = self.num_sc_per_rb * self.num_rb_per_subchannel * self.num_subchannel 

    def calculate_throughput_slot_subcarrier(self, channel_type):
        self.tx_array = PanelArray(num_rows_per_panel=1,
                                   num_cols_per_panel=1,
                                   polarization=self.polarization,
                                   polarization_type=self.polarization_type,
                                   antenna_pattern=self.antenna_pattern,
                                   carrier_frequency=self.frequency)
        
        self.rx_array = PanelArray(num_rows_per_panel=1,
                                   num_cols_per_panel=1,
                                   polarization=self.polarization,
                                   polarization_type=self.polarization_type,
                                   antenna_pattern=self.antenna_pattern,
                                   carrier_frequency=self.frequency)
        
        snr_total = self.calculate_snr(channel_type) # [num_frame, num_slot, num_subcarrier, num_veh * num_veh]
        tp_slot_subcarrier = self.scs * tf.math.log(1.0 + snr_total) / tf.math.log(tf.constant(2.0,dtype=tf.float32)) * self.time_slot_duration 
        tp_slot_subcarrier = tf.floor(tp_slot_subcarrier) # [num_frame, num_slot, num_subcarrier, v*v]
        return tp_slot_subcarrier
    
    # mode 1: time-frequency
    def calculate_throughput_slot_subchannel(self, tp_slot_subcarrier):
        tp_per_slot_carrier_reshape = tf.reshape(tp_slot_subcarrier,[self.num_frame,self.slot_per_frame,self.num_subchannel,
                                                                   self.num_sc_per_rb*self.num_rb_per_subchannel,self.num_vehicle,self.num_vehicle])
        tp_per_slot_subchannel = tf.reduce_sum(tp_per_slot_carrier_reshape, axis=3) # [num_frame, slot_per_frame, num_subchannel, num_veh , num_veh]
        np_tp_per_slot_subchannel = tp_per_slot_subchannel.numpy()
        return np_tp_per_slot_subchannel
    
    # mode 2: only time
    def calculate_throughput_slot_all_subchannel(self, tp_slot_subcarrier):
        tp_per_slot_carrier_reshape = tf.reshape(tp_slot_subcarrier,[self.num_frame,self.slot_per_frame,self.num_subchannel,
                                                                   self.num_sc_per_rb*self.num_rb_per_subchannel,self.num_vehicle,self.num_vehicle])
        tp_per_slot_subchannel = tf.reduce_sum(tp_per_slot_carrier_reshape, axis=3) # [num_frame, slot_per_frame, num_subchannel, num_veh , num_veh]
        tp_per_slot_all_subchannel = tf.reduce_sum(tp_per_slot_subchannel, axis=2) # [num_frame, slot_per_frame, num_veh , num_veh]
        np_tp_per_slot_all_subchannel = tp_per_slot_all_subchannel.numpy()
        return np_tp_per_slot_all_subchannel
    
    # mode 3: only frequency
    def calculate_throughput_subchannel_all_slot(self, tp_slot_subcarrier):
        tp_per_slot_carrier_reshape = tf.reshape(tp_slot_subcarrier,[self.num_frame,self.slot_per_frame,self.num_subchannel,
                                                                   self.num_sc_per_rb*self.num_rb_per_subchannel,self.num_vehicle,self.num_vehicle])
        tp_per_slot_subchannel = tf.reduce_sum(tp_per_slot_carrier_reshape, axis=3) # [num_frame, slot_per_frame, num_subchannel, num_veh , num_veh]
        np_tp_per_slot_subchanel= tp_per_slot_subchannel.numpy() # [num_frame, slot_per_frame, num_subchannel, num_veh , num_veh]
        return np_tp_per_slot_subchanel
    
    def calculate_snr(self, channel_type):
        tx_power_per_sc_dB = self._get_tx_power_per_sc() 
        no_power_per_sc_dB = self._get_no_power_per_sc(channel_type)

        lsf = LargeScaleFading(self.scenario, channel_type, self.frequency, self.distance, self.shadow_fading)
        large_scale_fading_dB = lsf.calculate_large_scale_fading()
        snr_dB = tx_power_per_sc_dB - no_power_per_sc_dB - large_scale_fading_dB

        snr_linear = db_to_lin(snr_dB) ##[num_frame, num_slot, num_veh, num_veh]
        snr_linear = tf.expand_dims(snr_linear, axis=2)
        snr_linear = tf.repeat(snr_linear,repeats=self.num_subcarrier,axis=2) #[num_frame, num_slot, num_subcarrier, num_veh, num_veh]

        if self.small_scale_fading:
            ssf = self._calculate_small_scale_fading(channel_type)
            ssf = tf.transpose(ssf, [1,2,3,0]) # [num_frame, num_slot, num_subcarrier, num_link=v(v-1)]
            snr_linear_ssf = self.apply_small_scale_fading(snr_linear, ssf) # [num_frame, num_slot, num_subcarrier, num_link=v(v-1)]
            snr_linear_ssf = reconstruct(snr_linear_ssf, self.num_vehicle) # [num_frame, num_slot, num_subcarrier, num_link=v * v]
            snr_clip = tf.minimum(snr_linear_ssf, tf.constant(db_to_lin(30)))
            return snr_clip
        snr_linear = tf.reshape(snr_linear, [self.num_frame,self.slot_per_frame,self.num_subcarrier,self.num_vehicle * self.num_vehicle])
        snr_clip = tf.minimum(snr_linear, tf.constant(db_to_lin(30)))
        return snr_clip

    def _get_tx_power_per_sc(self):
        num_sc = self.num_subcarrier
        total_tx_power_db = self.tx_power
        total_tx_power_li = db_to_lin(total_tx_power_db)
        tx_power_per_sc_li = total_tx_power_li / num_sc
        tx_power_per_sc_dB = lin_to_db(tx_power_per_sc_li)
        return tx_power_per_sc_dB
    
    def _get_no_power_per_sc(self, channel_type):
        K = 1.38 * 1e-23
        T = 293.15
        noise_power_linear = self.scs * K * T
        NF = 9 
        noise_power_db = lin_to_db(noise_power_linear) + 30 + NF

        if channel_type == "los" or channel_type=="nlosv":
            noise_power = db_to_lin(noise_power_db)
            Pi = db_to_lin(-72.25) / self.num_subcarrier # -72.25 for 20MHz
            noise_power_db = lin_to_db(noise_power + Pi)
        return noise_power_db
    
    def _parse_config(self):
        with open(self.config_path, 'r') as f:
            comm_config = yaml.load(f,Loader=loader)
        self.slot_per_frame = comm_config['slot_per_frame']
        self.effective_slot_per_frame = comm_config['effective_slot_per_frame']
        self.iter_times = comm_config['iter_times']
        self.num_partner = comm_config['num_partner'] 
        self.num_c = comm_config['num_c']
        self.bits_per_c = comm_config['bits_per_c']
        self.H = comm_config['H']
        self.W = comm_config['W']
        self.bandwidth = comm_config['bandwidth']
        self.scenario = comm_config['scenario']
        self.direction = comm_config['direction']
        self.frequency = comm_config['frequency']
        self.tx_power = comm_config['tx_power']
        self.num_subchannel = comm_config['num_subchannel']
        self.scs = comm_config['scs']
        self.time_slot_duration = comm_config['time_slot_duration']
        self.num_sc_per_rb = comm_config['num_sc_per_rb']
        self.num_rb_per_subchannel = comm_config['num_rb_per_subchannel']
        self.seed = comm_config['seed']
        self.polarization = comm_config['polarization']
        self.polarization_type = comm_config['polarization_type']
        self.antenna_pattern = comm_config['antenna_pattern']
        self.bits_per_grid = self.num_c * self.bits_per_c
        self.shadow_fading = comm_config['shadow_fading']
        self.small_scale_fading = comm_config['small_scale_fading']
        self.num_slot_per_group = NUM_SLOT_PER_GROUP

    
    def _set_seed(self):
        tf.random.set_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        sionna.phy.config.seed = self.seed
    

    def _calculate_small_scale_fading(self, channel_type):
        tx_array = self.tx_array
        rx_array = self.rx_array
        model = self.scenario + "_" + channel_type
        direction = self.direction
        delay_spread = 1e-9
        cdl = CDL(model=model, delay_spread=delay_spread, carrier_frequency=self.frequency, tx_array=tx_array, rx_array=rx_array, direction=direction)
        slot_total = self.slot_per_frame * self.num_frame
        num_links = self.num_vehicle * (self.num_vehicle - 1) 
        a, tau = cdl(batch_size=num_links,num_time_steps=slot_total,sampling_frequency=1000)
        sub_freq = subcarrier_frequencies(num_subcarriers=self.num_subcarrier,subcarrier_spacing=self.scs)
        h_freq = cir_to_ofdm_channel(frequencies=sub_freq,
                                     a=a,
                                     tau=tau)
        h_freq = tf.squeeze(h_freq)  # [num_link=v(v-1),num_slot_total, num_subcarrier]
        ssf = tf.abs(h_freq) ** 2   # [num_link=v(v-1),num_slot_total, num_subcarrier]
        ssf = tf.reshape(ssf, shape=[num_links,self.num_frame, self.slot_per_frame,self.num_subcarrier]) #[num_link, num_frame, num_slot, num_subcarrier]
        return ssf
    
    def apply_small_scale_fading(self, snr_linear, ssf):
        i_indices = tf.range(self.num_vehicle, dtype=tf.int32)
        j_indices = tf.range(self.num_vehicle, dtype=tf.int32)
        I, J = tf.meshgrid(i_indices, j_indices, indexing='ij')  # [V, V]
        mask = tf.not_equal(I, J)
        snr_linear_flat = tf.reshape(snr_linear,[self.num_frame,self.slot_per_frame,self.num_subcarrier,-1])  # [num_frame, num_slot, num_subcarrier, num_veh*num_veh]
        mask_flat = tf.reshape(mask,shape=[-1])  # [vehicle * vehicle]
        snr_transposed = tf.transpose(snr_linear_flat, perm=[3, 0, 1, 2])  # [V*V, F, S, K]
        snr_valid = tf.boolean_mask(snr_transposed, mask_flat)      # [v*v-1, F, S, K]
        snr_valid = tf.transpose(snr_valid, perm=[1, 2, 3, 0])      # [F, S, K, v*(v-1)]
        snr_flat = snr_valid * ssf  # [F, S, K, v*(v-1)]
        return snr_flat
    
    def merge_tp(self, tp_los, tp_nlos, tp_nlosv, mode=1):

        if 1 == mode: # [num_frame, slot_per_frame, num_subchannel, num_veh , num_veh]

            distance, nlos_mask, _, _ = build_vehicle_distance_tensor(self.distance_path)
            p_los = tf.minimum(1.0, 1.05 * tf.exp(-0.0114 * distance)).numpy()
            p_los[np.arange(self.num_frame)[:, None], np.arange(self.num_vehicle), np.arange(self.num_vehicle)] = 0.0

            random_r = np.random.rand(self.num_frame, self.num_vehicle, self.num_vehicle)
            mask = (random_r <= p_los).astype(np.float32) 

            mask_sym = np.maximum(mask, mask.transpose(0, 2, 1))
            mask_sym[np.arange(self.num_frame)[:, None], np.arange(self.num_vehicle), np.arange(self.num_vehicle)] = 0.0

            mask_sym_expand = np.expand_dims(mask_sym, axis=1)
            mask_sym_expand = np.repeat(mask_sym_expand, repeats=self.slot_per_frame, axis=1)

            mask_sym_expand_expand = np.expand_dims(mask_sym_expand, axis=2)
            mask_sym_expand_expand = np.repeat(mask_sym_expand_expand, repeats=self.num_subchannel, axis=2)

            tp_prob_select = mask_sym_expand_expand * tp_los + (1 - mask_sym_expand_expand) * tp_nlosv

            nlos_mask_expand = np.expand_dims(nlos_mask.numpy(), axis=1)
            nlos_mask_expand = np.repeat(nlos_mask_expand, repeats=self.slot_per_frame, axis=1)

            nlos_mask_expand_expand = np.expand_dims(nlos_mask_expand, axis=2)
            nlos_mask_expand_expand = np.repeat(nlos_mask_expand_expand, repeats=self.num_subchannel, axis=2)

            tp_slot_subchannel = np.zeros_like(tp_los, dtype=np.float32)
            tp_slot_subchannel[nlos_mask_expand_expand == 1.0] = tp_nlos[nlos_mask_expand_expand == 1.0]
            tp_slot_subchannel[nlos_mask_expand_expand == 0.0] = tp_prob_select[nlos_mask_expand_expand == 0.0]


            tp_Nslots_subchannel = np.reshape(tp_slot_subchannel[:,:self.effective_slot_per_frame,:,:,:], 
                                                 [self.num_frame, int(self.effective_slot_per_frame/NUM_SLOT_PER_GROUP), NUM_SLOT_PER_GROUP, 
                                                  self.num_subchannel, self.num_vehicle, self.num_vehicle])
            tp_Nslots_subchannel = np.sum(tp_Nslots_subchannel,axis=2) # [F, T/10, SCH,V, V]
            tp_Nslots_subchannel = np.floor(tp_Nslots_subchannel * ALPHA)

            return tp_Nslots_subchannel

        elif 2 == mode: # [num_frame, slot_per_frame, num_veh , num_veh]

            distance, nlos_mask, _, _ = build_vehicle_distance_tensor(self.distance_path)
            p_los = tf.minimum(1.0, 1.05 * tf.exp(-0.0114 * distance)).numpy()
            p_los[np.arange(self.num_frame)[:, None], np.arange(self.num_vehicle), np.arange(self.num_vehicle)] = 0.0
            random_r = np.random.rand(self.num_frame, self.num_vehicle, self.num_vehicle)
            mask = (random_r <= p_los).astype(np.float32) 

            mask_sym = np.maximum(mask, mask.transpose(0, 2, 1))
            mask_sym[np.arange(self.num_frame)[:, None], np.arange(self.num_vehicle), np.arange(self.num_vehicle)] = 0.0

            mask_sym_expand = np.expand_dims(mask_sym, axis=1) 
            mask_sym_expand = np.repeat(mask_sym_expand, repeats=self.slot_per_frame, axis=1)  

            tp_prob_select = mask_sym_expand * tp_los + (1 - mask_sym_expand) * tp_nlosv

            #[F, V, V] → [F, T, V, V]
            nlos_mask_expand = np.expand_dims(nlos_mask.numpy(), axis=1)
            nlos_mask_expand = np.repeat(nlos_mask_expand, repeats=self.slot_per_frame, axis=1)

            tp_all_subchannel = np.zeros_like(tp_los, dtype=np.float32)
            tp_all_subchannel[nlos_mask_expand == 1.0] = tp_nlos[nlos_mask_expand == 1.0]
            tp_all_subchannel[nlos_mask_expand == 0.0] = tp_prob_select[nlos_mask_expand == 0.0]

            tp_all_subchannel = np.floor(tp_all_subchannel * ALPHA)
            
            return tp_all_subchannel
        
        elif 3 == mode: # [num_frame, slot_per_frame, num_subchannel, num_veh , num_veh]

            distance, nlos_mask, _, _ = build_vehicle_distance_tensor(self.distance_path)   
            p_los = tf.minimum(1.0, 1.05 * tf.exp(-0.0114 * distance)).numpy()
            p_los[np.arange(self.num_frame)[:, None], np.arange(self.num_vehicle), np.arange(self.num_vehicle)] = 0.0

            random_r = np.random.rand(self.num_frame, self.num_vehicle, self.num_vehicle)
            mask = (random_r <= p_los).astype(np.float32)
            mask_sym = np.maximum(mask, mask.transpose(0, 2, 1))
            mask_sym[np.arange(self.num_frame)[:, None], np.arange(self.num_vehicle), np.arange(self.num_vehicle)] = 0.0

            mask_sym_expand = np.expand_dims(mask_sym, axis=1)
            mask_sym_expand = np.repeat(mask_sym_expand, repeats=self.slot_per_frame, axis=1)
            mask_sym_expand_expand = np.expand_dims(mask_sym_expand, axis=2)
            mask_sym_expand_expand = np.repeat(mask_sym_expand_expand, repeats=self.num_subchannel, axis=2)

            tp_prob_select = mask_sym_expand_expand * tp_los + (1 - mask_sym_expand_expand) * tp_nlosv

            nlos_mask_expand = np.expand_dims(nlos_mask.numpy(), axis=1)
            nlos_mask_expand = np.repeat(nlos_mask_expand, repeats=self.slot_per_frame, axis=1)
            nlos_mask_expand_expand = np.expand_dims(nlos_mask_expand, axis=2)
            nlos_mask_expand_expand = np.repeat(nlos_mask_expand_expand, repeats=self.num_subchannel, axis=2)

            tp_slot_subchannel = np.zeros_like(tp_los, dtype=np.float32)
            tp_slot_subchannel[nlos_mask_expand_expand == 1.0] = tp_nlos[nlos_mask_expand_expand == 1.0]
            tp_slot_subchannel[nlos_mask_expand_expand == 0.0] = tp_prob_select[nlos_mask_expand_expand == 0.0]

            tp_all_slot = np.sum(tp_slot_subchannel[:,:self.effective_slot_per_frame,:,:,:], axis=1) #  [num_frame, num_subchannel, num_veh , num_veh]
            tp_all_slot = np.floor(tp_all_slot * ALPHA)
            return tp_all_slot


if __name__ == "__main__":
    comm_dir = "/mnt/data/Sionna_V2V/comm/comm_config.yaml"
    a = CalculateThroughput(comm_dir, "/home/intel/Gao/OPV2V/T/2021_08_24_11_37_54/")
    a.calculate_throughput_slot_subcarrier("los")
