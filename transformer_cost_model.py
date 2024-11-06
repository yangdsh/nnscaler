# For transformer

class CostModel:
    
    def __init__(self):
        # sharding spec
        self.dp = 8
        self.tp = 2
        self.cp = 2
        self.pp = 8
        self.l = 2 # number of layers per stage
        self.offload = 0
        
        # model spec
        self.V = 51200
        self.h = 8192
        self.H = 28672
        self.a = 64 # number of q heads
        self.g = 8 # number of kv heads
        self.s = 8192
        self.L = 80
        
        # gpu spec
        self.gpu_mem = 65 * 10**9 # 40GB
        
        # profiling
        self.comp = 150 * 10**12
        self.comm = 200 * 10**9

    # memory cost model
    def memory_bytes(self):
        p = (2+ 2 * self.g/self.a + 2 * self.H/self.h) * self.h * self.h
        l_ = self.L / self.pp
        
        weight_gradient = (2+2) / self.tp * \
            (l_ * p + self.V * self.h) # can be optimized
        
        optim = 12 / self.tp / self.cp / self.dp * \
            (l_ * p + self.V * self.h)
        
        # flash attention
        act = 1 / self.tp / self.cp * \
            (12 + 4 * self.g / self.a + 8 * self.H / self.h) * self.l * self.s * self.h
        
        act_ckpt = 1 / self.tp / self.cp * \
            (8 + 4 * self.g / self.a + 4 * self.H / self.h) * self.l * self.s * self.h
        
        v = self.L / self.pp / self.l
        act_gpu = ((v * self.pp + self.pp - 3) * (1 - self.offload) + 2 + 2 * self.offload) * act_ckpt
        
        print(f'weight_gradient: {weight_gradient / 10**9}, optim: {optim / 10**9}, act_gpu: {act_gpu / 10**9}')
        mem_gpu = weight_gradient + optim + act_gpu
        return mem_gpu
        
    # time cost model
    def compute_bytes(self):
        attention = (2 + 2 + (2+2) * self.g / self.a) * self.s * self.h * self.h +\
            (2 + 2) * self.s * self.s * self.h 
        attention = attention / self.tp / self.cp
        feed_forward = 2 * self.s * self.h * self.H / self.tp / self.cp
        return attention + feed_forward
    
    def comm_bytes(self):
        tp_2ag_2rs = 2 * (2 + 2 + 4 + 4) * self.s * self.h / self.cp
        cp_2ag_2rs_1ag = 12 * self.g / self.a * self.s * self.h / self.tp
        if self.tp == 1:
            tp_2ag_2rs = 0
        if self.cp == 1:
            cp_2ag_2rs_1ag = 0
        return tp_2ag_2rs + cp_2ag_2rs_1ag
    
    def search(self):
        t = self.compute_bytes() / self.comp
        c = self.comm_bytes() / self.comm
        m = self.memory_bytes() / self.gpu_mem
        print(f'compute: {t}, comm: {c}, mem: {m}')
    
    def offload_ovehead(self):
        pass


CostModel().search()
