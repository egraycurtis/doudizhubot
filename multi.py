import multiprocessing
from self_play import self_play

def train_parallel(partition):
    try:
        self_play(partition)
    except Exception as e:
        print(f"Error in process: {e}")

def main():
    cpu_count = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=cpu_count) as pool:
        pool.map(train_parallel, range(cpu_count))

if __name__ == "__main__":
    main()