class Config:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
            
def test_config_init():
    c = Config(name='test', data=[1,2,3])
    print(dir(c))
    
if __name__ == '__main__':
    test_config_init()