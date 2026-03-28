from jarvis.arm.models.wrappers.lightning_wrapper import (
    LightningBase, 
    NaiveTrainingMixin, TBPTTMethodMixin, 
    BehaviorCloningMixin, DirectPreferenceLearningMixin, 
)

class NaiveBehaviorCloning(LightningBase, NaiveTrainingMixin, BehaviorCloningMixin):
    
    def __init__(self, *args, **kwargs):
        LightningBase.__init__(self, *args, **kwargs)
        NaiveTrainingMixin.__init__(self)
        BehaviorCloningMixin.__init__(self)

class TBPTTBehaviorCloning(LightningBase, TBPTTMethodMixin, BehaviorCloningMixin):
    
    def __init__(self, *args, **kwargs):
        LightningBase.__init__(self, *args, **kwargs)
        TBPTTMethodMixin.__init__(self)
        BehaviorCloningMixin.__init__(self)

class NaiveDirectPreferenceLearning(LightningBase, NaiveTrainingMixin, DirectPreferenceLearningMixin):
    
    def __init__(self, *args, **kwargs):
        LightningBase.__init__(self, *args, **kwargs)
        NaiveTrainingMixin.__init__(self)
        DirectPreferenceLearningMixin.__init__(self, *args, **kwargs)


class TBPTTDirectPreferenceLearning(LightningBase, TBPTTMethodMixin, DirectPreferenceLearningMixin):
    
    def __init__(self, *args, **kwargs):
        LightningBase.__init__(self, *args, **kwargs)
        TBPTTMethodMixin.__init__(self)
        DirectPreferenceLearningMixin.__init__(self, *args, **kwargs)