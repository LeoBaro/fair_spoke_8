import ray

@ray.remote
class Worker:

    def train(self, data_iterator):
        for batch in data_iterator.iter_batches(batch_size=8):
            pass

ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")
workers = [Worker.remote() for _ in range(4)]

shards = ds.streaming_split(n=4, equal=True)
ray.get([w.train.remote(s) for w, s in zip(workers, shards)])