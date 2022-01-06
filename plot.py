import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv("measurements.csv", sep=',')
    print(data)
    plt.plot(data['Resolution'], data['CPU'], label='CPU times')
    plt.plot(data['Resolution'], data['GPU'], label='GPU times')
    plt.xlabel('Resolution')
    plt.ylabel('Time (microseconds)')
    plt.title('Runtime Comparison (Jacobi Solver)')
    plt.legend()
    plt.savefig('plot_jacobi.png')