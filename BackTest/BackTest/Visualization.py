import matplotlib.pyplot as plt
import seaborn as sns

def show(excess, benchmark, cum_pnl, cum_exc_pnl, benchmark_return):
    if excess:
        plt.plot(cum_exc_pnl)
        legend = ['cum_exc_pnl']
    else:
        legend = []
        fig = plt.figure()
        plt.plot(cum_pnl)
        legend.append('cum_pnl')
        if benchmark:
            plt.plot((1+benchmark_return).cumprod())
            legend.append('benchmark')
    plt.legend(legend)
    print(legend)
    plt.show()

def LnS_Pnl_show(excess, benchmark, tuple_of_list, benchmark_return):
    groups = len(tuple_of_list[0])
    if excess: 
        figure = plt.figure()
        no_stocks_groups = []
        for group in range(groups):
            if tuple_of_list[3][group] is None:
                no_stocks_groups.append(group+1)
            else:
                plt.plot((tuple_of_list[3][group] + 1).cumprod())

        list_1 = list(range(1, groups+1))
        legend_list = [item for item in list_1 if item not in no_stocks_groups]      
        plt.legend(legend_list)
        plt.show()
        print(f'group_1 is the largest short and group_{groups} is the largest long')
        for group in range(groups):
            print(f'the R-squared of the group_{group+1} is: {tuple_of_list[1][group]}')
    else:
        figure = plt.figure()
        no_stocks_groups = []
        for group in range(groups):
            if tuple_of_list[2][group] is None:
                no_stocks_groups.append(group + 1)
            else:
                plt.plot((tuple_of_list[2][group] + 1).cumprod())
        list_1 = list(range(1, groups+1))
        legend_list = [item for item in list_1 if item not in no_stocks_groups]
        if benchmark:
            plt.plot((1+benchmark_return).cumprod())
            legend_list.append('benchmark')
        plt.legend(legend_list)
        plt.show()
        print(f'group_1 is the largest short and group_{groups} is the largest long')
        for group in range(groups):
            print(f'the R-squared of the group_{group+1} is: {tuple_of_list[0][group]}')
    return 0

def show_corr(corr_matrix):
    sns.set(style="white")
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title("Correlation Matrix Heatmap")
    plt.show()