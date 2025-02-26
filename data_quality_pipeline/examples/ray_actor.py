import ray
ray.init()

@ray.remote
class Worker(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def read(self):
        return self.n

workers = [Worker.remote() for i in range(4)]
[w.increment.remote() for w in workers]
futures = [w.read.remote() for w in workers]

print(ray.get(futures)) # [1, 1, 1, 1]