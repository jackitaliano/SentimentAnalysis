import seaborn as sns
import matplotlib.pyplot as plt


class DataAnalysis:
    '''Used to analyze data before and after ML to get a sense of the data'''
    def plot_dist_labels(labels: list, class_names: list[str]) -> None:
        '''Plot distribution of labels over class names'''
        DataAnalysis.plot_count_over_classes(
            labels, class_names, 'Class Distribution')

    def plot_dist_lens(df, tokenizer) -> None:
        '''Plot distribution of lens of sequences in df.content'''
        token_lens = []
        for txt in df.content:
            tokens = tokenizer.encode(txt, max_length=512, truncation=True)
            token_lens.append(len(tokens))

        DataAnalysis.plot_dist_over_range(token_lens, [1, 256], 'Token Count')

    def plot_count_over_classes(data: list, class_names: list[str], x_label: str) -> None:
        sns.set()
        ax = sns.countplot(x=data)
        plt.xlabel(x_label)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names)

        plt.show()

    def plot_dist_over_range(data: list[int], d_range: list[int], x_label: str) -> None:
        sns.displot(data, kde=True)
        plt.xlim(d_range)
        plt.xlabel(x_label)

        plt.show()
