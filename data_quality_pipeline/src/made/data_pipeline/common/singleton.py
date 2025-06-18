class Singleton(type):  
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
    @classmethod
    def destroy_instance(cls, target_cls):
        """Destroy the singleton instance of the specified class."""
        if target_cls in cls._instances:
            del cls._instances[target_cls]