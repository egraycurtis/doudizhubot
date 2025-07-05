import multiprocessing
from compete import compete
from dbsetup import setup_database
from self_play import self_play
from train import train


if __name__ == "__main__":
    setup_database()
    training_models = ['transformer']
    cpu_count = multiprocessing.cpu_count()
    tasks = [(i, training_models[i%len(training_models)]) for i in range(cpu_count - 2)]

    with multiprocessing.Pool(processes=cpu_count) as pool:
        pool.starmap_async(self_play, tasks)
        pool.apply_async(train)
        pool.apply_async(compete, (['deep', 'deep', 'transformer', 'transformer'],))
        
        pool.close()
        pool.join()