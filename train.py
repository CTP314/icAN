class trainer():
    def __init__(self, config):
        # factor for balancing different losses
        self.L1_penalty = config.L1_penalty
        self.Lconst_penalty = config.Lconst_penalty
        self.Ltv_penalty = config.Ltv_penalty
        self.Lcategory_penalty = config.Lcategory_penalty
    
    