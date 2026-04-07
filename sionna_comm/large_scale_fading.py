import tensorflow as tf


class LargeScaleFading():
    def __init__(self, scenario, channel, frequency, distance, shadow_fading):
        self.scenario = scenario
        self.channel = channel
        self.frequency = frequency
        self.distance = distance  
        self.shape = tf.shape(distance)
        self.shadow_fading = shadow_fading
        assert scenario == "urban"
        assert channel in ["los", "nlos", "nlosv"]

    def calculate_large_scale_fading(self):
        if self.shadow_fading:
            lsf = tf.constant(0.0, dtype=tf.float32, shape=self.shape)
            pl = self._calculate_pathloss(self.channel, self.frequency, self.distance)
            sf = self._calculate_shadow_fading(self.channel)
            bl = self._calculate_blockage_loss()
            lsf += (pl + sf + bl)
            return lsf
        else:
            lsf = tf.constant(0.0, dtype=tf.float32, shape=self.shape)
            pl = self._calculate_pathloss(self.channel, self.frequency, self.distance)
            bl = self._calculate_blockage_loss()
            lsf += (pl + bl)
            return lsf
    
    def _calculate_pathloss(self, channel, frequency, distance):
        min_distance =tf.constant(10.0, dtype=tf.float32)
        clip_distance = tf.maximum(min_distance,distance)
        frequency /= 1e9
        if channel == "los":
            pl = 38.77 + 16.7 * tf.math.log(clip_distance) / tf.math.log(tf.constant(10.0, dtype=tf.float32)) + 18.2 * tf.math.log(frequency) / tf.math.log(tf.constant(10.0,dtype=tf.float32))
            return pl
        elif channel == "nlosv":
            pl = 38.77 + 16.7 * tf.math.log(clip_distance) / tf.math.log(tf.constant(10.0, dtype=tf.float32)) + 18.2 * tf.math.log(frequency) / tf.math.log(tf.constant(10.0,dtype=tf.float32))
            return pl
        elif channel == "nlos":
            pl = 36.74 + 30 * tf.math.log(clip_distance) / tf.math.log(tf.constant(10.0, dtype=tf.float32)) + 18.9 * tf.math.log(frequency) / tf.math.log(tf.constant(10.0,dtype=tf.float32))
            return pl
    
    def _calculate_shadow_fading(self, channel):
        if channel in ["los", "nlosv"]:
            std = (tf.constant(3.0,dtype=tf.float32))
        if channel == "nlos":
            std = (tf.constant(4.0,dtype=tf.float32))
        veh_indices = tf.range(tf.shape(self.distance)[-1], dtype=tf.int32)
        i, j = tf.meshgrid(veh_indices, veh_indices, indexing='ij') 
        mask_upper = tf.cast(i < j, tf.float32) 
        mask_diag  = tf.cast(i == j, tf.float32) 
        sf_random = tf.random.normal(shape=self.shape, mean=0.0, stddev=std, dtype=tf.float32)
        sf_upper  = sf_random * mask_upper 
        sf_upper_t = tf.transpose(sf_upper, perm=[0,1,3,2])
        sf_db = sf_upper + sf_upper_t 
        sf_db = sf_db * (1.0 - mask_diag)
        return sf_db
    
    def _calculate_blockage_loss(self):
        if self.channel == "nlosv":
        # assume that height value of TX or RX equals to Blocker height
            mean = 5 + tf.maximum(0, 15 * tf.math.log(self.distance) / tf.math.log(tf.constant(10.0, dtype=tf.float32)) - 41)
            std = 4
            blockage_loss = tf.random.normal(shape=self.shape, mean=mean,stddev=std,dtype=tf.float32)
        else:
            blockage_loss = tf.constant(0.0, dtype=tf.float32, shape=self.shape)
        return blockage_loss

