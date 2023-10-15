from pathlib import Path
import os
import time

import numpy as np
import pandas as pd

from constants import SEEDS
import matplotlib.pyplot as plt


class ChartProducer(object):
    def __init__(self, problem_name, problem_sizes, maximise, max_fitness, iterations=300):
        self.algo_parameters = {
            'ga': ['Mutation Rate'],
            'mimic': ['Keep Percent'],
            'rhc': [],
            'sa': ['Temperature']
        }
        assert len(problem_sizes) == len(max_fitness)
        self.max_fitness = dict(list(zip(problem_sizes, max_fitness)))
        self.iterations = iterations
        self.problem_name = problem_name
        self.problem_sizes = problem_sizes
        self.maximise = maximise
        self.curve_dfs = {}
        self.stats_dfs = {}
        ChartProducer.setup_plots()
        for size in self.problem_sizes:
            for algo in self.algo_parameters.keys():
                stats_df = self._get_run_stats_dataset(size, algo)
                curve_df = self._get_curves_dataset(size, algo)
                self.stats_dfs[f'{size}-{algo}'] = stats_df
                self.curve_dfs[f'{size}-{algo}'] = curve_df
        print(f"All datasets loaded!")

    @classmethod
    def setup_plots(cls):
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['figure.figsize'] = (8, 6)

    def plot_all_series_for_all(self, curve='Fitness'):
        for size in self.problem_sizes:
            for algo in self.algo_parameters:
                self._plot_curves_against_iterations_single(size, algo, curve)

    def plot_best_series_for_all_sizes(self, curve='Fitness'):
        for size in self.problem_sizes:
            best_curves_dfs = self.get_best_curves_by_fitness_algo(size)
            best_curves_stacked_df = pd.concat(best_curves_dfs, axis=0)
            self._plot_curve(df=best_curves_stacked_df,
                             curve=curve,
                             title_details=f'Best Parameters, (size={size})',
                             file_name=f'charts/{self.problem_name}-{size}/{curve.lower()}-best.png',
                             breakdown_column='Algorithm'
                             )

    def plot_best_series_for_all_algos(self, curve='Fitness'):
        for algo in self.algo_parameters.keys():
            best_curves_dfs = self.get_best_results_by_fitness_size(algo)
            best_curves_stacked_df = pd.concat(best_curves_dfs, axis=0)
            self._plot_barchart(df=best_curves_stacked_df,
                                curve=curve,
                                title_details=f'Best Parameters ({algo.upper()})',
                                file_name=f'charts/{self.problem_name}-{algo}/{curve.lower()}-best.png',
                                breakdown_column='Problem Size'
                                )

    def fill_in(self, df_seed, algo):
        m = self.iterations + 1
        df_w = df_seed.drop(columns=['Unnamed: 0']).reset_index(drop=True).copy()

        def helper(this_df):
            n = this_df['Iteration'].max()
            num_duplicates_needed = max(0, m - len(this_df))
            highest_iteration_row = this_df[this_df['Iteration'] == n].copy()
            copied_rows_list = [highest_iteration_row.copy() for _ in range(num_duplicates_needed)]
            copied_rows = pd.concat(copied_rows_list)
            copied_rows['Iteration'] = range(n + 1, n + 1 + num_duplicates_needed)
            return pd.concat([this_df, copied_rows], ignore_index=True)

        # we assume we tune 1 parameter max
        if len(self.algo_parameters[algo]) == 0:
            if len(df_w) >= m:
                return df_w
            padded_df = helper(df_w)
            return padded_df.sort_values(['Iteration'])

        param_name = self.algo_parameters[algo][0]
        param_values = df_w[param_name].unique()
        result_slices = []
        for param_value in param_values:
            cur_slice = df_w[df_w[param_name] == param_value].reset_index(drop=True).copy()
            add = None
            for i in range(1, len(cur_slice)):
                if cur_slice.loc[i, 'Time'] <= 0.1 * cur_slice.loc[i - 1, 'Time'] and add is None:
                    add = cur_slice.loc[i - 1, 'Time']
                # Adjust the threshold as needed
                if add is not None:
                    cur_slice.loc[i, 'Time'] = cur_slice.loc[i, 'Time'] + add

            if len(cur_slice) >= m:
                result_slices.append(cur_slice.copy())
            else:
                padded_slice = helper(cur_slice)
                result_slices.append(padded_slice)
        return pd.concat(result_slices).sort_values(by=[param_name, 'Iteration'], ascending=True).reset_index(drop=True)

    def _get_curves_dataset(self, size, algo):
        file_names = [
            f'outputs/{self.problem_name}-{size}/{algo}-{self.problem_name}-{size}-seed-{i}/{algo}__{algo}-{self.problem_name}-{size}-seed-{i}__curves_df.csv'
            for
            i in range(len(SEEDS))]
        dfs = [pd.read_csv(file) for file in file_names]
        dfs = [self.fill_in(df, algo) for df in dfs]
        keep_columns = ['Iteration'] + self.algo_parameters[algo]

        # Merge the dataframes based on the keep columns
        result_df = pd.concat(dfs, axis=1)[keep_columns]
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        result_df['Mean Fitness'] = pd.concat([df['Fitness'] for df in dfs], axis=1).mean(axis=1)
        result_df['Std Fitness'] = pd.concat([df['Fitness'] for df in dfs], axis=1).std(axis=1)
        result_df['Mean Time'] = pd.concat([df['Time'] for df in dfs], axis=1).mean(axis=1)
        result_df['Std Time'] = pd.concat([df['Time'] for df in dfs], axis=1).std(axis=1)
        result_df['Mean FEvals'] = pd.concat([df['FEvals'] for df in dfs], axis=1).mean(axis=1)
        result_df['Std FEvals'] = pd.concat([df['FEvals'] for df in dfs], axis=1).std(axis=1)

        return result_df

    def get_best_results_by_fitness_size(self, algo):
        dfs = []
        for size in self.problem_sizes:
            df = self.get_best_curves_by_fitness_common(size, algo)
            dfs.append(df)
        return dfs

    def get_best_curves_by_fitness_algo(self, size):
        dfs = []
        for algo in self.algo_parameters.keys():
            df = self.get_best_curves_by_fitness_common(size, algo)
            dfs.append(df)
        return dfs

    def get_best_curves_by_fitness_common(self, size, algo):
        best_params_dict = self._get_best_parameters_by_fitness(size)
        best_params = best_params_dict[f'{size}-{algo}']
        params_names = self.algo_parameters[algo]
        cur_curve_df = self.curve_dfs[f'{size}-{algo}']
        filter_dict = dict(zip(params_names, best_params))
        filtered_df = cur_curve_df[
            cur_curve_df[list(filter_dict.keys())].eq(list(filter_dict.values())).all(axis=1)]
        filtered_df['Algorithm'] = algo.upper()
        filtered_df['Problem Size'] = size
        return filtered_df

    def _get_best_parameters_by_fitness(self, size):
        best_params_by_size_fitness = {}
        for algo in self.algo_parameters.keys():
            stats_df = self.stats_dfs[f'{size}-{algo}']
            stats_df = stats_df[stats_df['Iteration'] != 0]
            sorted_stats_df = stats_df.sort_values(by='Mean Fitness', ascending=not self.maximise)
            best_run = sorted_stats_df.iloc[0]
            param_values = []
            for param in self.algo_parameters[algo]:
                param_values.append(best_run[param])

            best_params_by_size_fitness[f'{size}-{algo}'] = param_values

        return best_params_by_size_fitness

    def _get_run_stats_dataset(self, size, algo):
        file_names = [
            f'outputs/{self.problem_name}-{size}/{algo}-{self.problem_name}-{size}-seed-{i}/{algo}__{algo}-{self.problem_name}-{size}-seed-{i}__run_stats_df.csv'
            for
            i in range(len(SEEDS))]
        dfs = [pd.read_csv(file) for file in file_names]
        keep_columns = ['Iteration'] + self.algo_parameters[algo]

        # Merge the dataframes based on the keep columns
        result_df = pd.concat(dfs, axis=1)[keep_columns]
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        result_df['Mean Fitness'] = pd.concat([df['Fitness'] for df in dfs], axis=1).mean(axis=1)
        result_df['Std Fitness'] = pd.concat([df['Fitness'] for df in dfs], axis=1).std(axis=1)
        result_df['Mean Time'] = pd.concat([df['Time'] for df in dfs], axis=1).mean(axis=1)
        result_df['Std Time'] = pd.concat([df['Time'] for df in dfs], axis=1).std(axis=1)
        result_df['Mean FEvals'] = pd.concat([df['FEvals'] for df in dfs], axis=1).mean(axis=1)
        result_df['Std FEvals'] = pd.concat([df['FEvals'] for df in dfs], axis=1).std(axis=1)

        return result_df

    def _plot_curves_against_iterations_single(self, size: int, algo: str, curve: str = 'Fitness'):
        df = self.curve_dfs[f'{size}-{algo}']

        if self.algo_parameters[algo] is None or len(self.algo_parameters[algo]) == 0:
            self._plot_curve(df=df,
                             curve=curve,
                             title_details=f'size={size} ({algo.upper()})',
                             file_name=f'charts/{self.problem_name}-{size}/{curve.lower()}-{algo}.png',
                             breakdown_column=None,
                             )
        elif len(self.algo_parameters[algo]) > 2:
            print('Not supported number of parameters')
        elif len(self.algo_parameters[algo]) == 1:
            breakdown_column = self.algo_parameters[algo][0]
            self._plot_curve(df=df,
                             curve=curve,
                             title_details=f'size={size} ({algo.upper()})',
                             file_name=f'charts/{self.problem_name}-{size}/{curve.lower()}-{algo}.png',
                             breakdown_column=breakdown_column,
                             )
        elif len(self.algo_parameters[algo]) == 2:
            breakdown_column_1 = self.algo_parameters[algo][0]
            breakdowns_1 = df[breakdown_column_1].unique()
            breakdown_column_2 = self.algo_parameters[algo][1]
            breakdowns_2 = df[breakdown_column_2].unique()

            for breakdown_1 in breakdowns_1:
                cur_df = df[df[breakdown_column_1] == breakdown_1]
                self._plot_curve(df=cur_df,
                                 curve=curve,
                                 title_details=f'size={size} ({algo.upper()}, {breakdown_column_1} = {breakdown_1})',
                                 file_name=f'charts/{self.problem_name}-{size}/{curve.lower()}-{algo}-{breakdown_column_1.lower().replace(" ", "-")}-{breakdown_1}.png',
                                 breakdown_column=breakdown_column_2,
                                 )

            for breakdown_2 in breakdowns_2:
                cur_df = df[df[breakdown_column_2] == breakdown_2]
                self._plot_curve(df=cur_df,
                                 curve=curve,
                                 title_details=f'size={size} ({algo.upper()}, {breakdown_column_2} = {breakdown_2})',
                                 file_name=f'charts/{self.problem_name}-{size}/{curve.lower()}-{algo}-{breakdown_column_2.lower().replace(" ", "-")}-{breakdown_2}.png',
                                 breakdown_column=breakdown_column_2,
                                 )

    def _plot_barchart(self, df, curve, title_details, file_name, breakdown_column):
        alpha = 0.2
        xs = df[breakdown_column].unique()
        ys = []
        ys_std = []

        for x in xs:
            y = df[(df[breakdown_column] == x) & (df['Iteration'] == self.iterations)]
            if curve == 'Fitness':
                max_fitness = self.max_fitness[x]
                std = y[f'Std {curve}'].iloc[0] / (max_fitness + 0.0)
                if self.maximise:
                    val = (y[f'Mean {curve}'].iloc[0] + 0.0) / (max_fitness + 0.0)

                else:
                    val = 1 - (y[f'Mean {curve}'].iloc[0] + 0.0) / (max_fitness + 0.0)
            else:
                val = y[f'Mean {curve}'].iloc[0]
                std = y[f'Std {curve}'].iloc[0]

            ys.append(val)
            ys_std.append(std)
        # ys_m_std = [y - y_std for y, y_std in zip(ys, ys_std)]
        # ys_p_std = [y + y_std for y, y_std in zip(ys, ys_std)]
        x_axis = np.arange(len(xs))
        plt.bar(x_axis, ys,
                # label=f'{curve} vs {breakdown_column}', linestyle='dotted'
                )
        # plt.scatter(x_axis, ys, s=10)
        plt.errorbar(x_axis, ys, yerr=ys_std, fmt="o", color="r")
        # plt.fill_between(x_axis,
        #                  ys_m_std,
        #                  ys_p_std, alpha=alpha)

        title = f'{curve} vs. {breakdown_column}'
        if title_details is not None:
            title = title + f', {title_details}'

        y_label_text = 'Fitness, %' if curve == 'Fitness' else f'{curve}'
        plt.xticks(x_axis, xs)
        plt.xlabel(f'{breakdown_column}')
        plt.ylabel(y_label_text)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        os.makedirs(Path(file_name).parent.absolute(), exist_ok=True)
        plt.savefig(file_name)
        plt.clf()

    def _plot_curve(self, df, curve, title_details, file_name, breakdown_column):
        alpha = 0.2
        min_val = np.inf
        max_val = -1 * np.inf
        if not self.maximise and curve == 'Fitness':
            multiplier = -1
        else:
            multiplier = 1
        if breakdown_column is None:
            plt.plot(df['Iteration'], multiplier * df[f'Mean {curve}'],
                     label=curve)
            plt.fill_between(df['Iteration'],
                             multiplier * df[f'Mean {curve}'] - df[f'Std {curve}'],
                             multiplier * df[f'Mean {curve}'] + df[f'Std {curve}'], alpha=alpha)
            max_val = max(np.max(multiplier * df[f'Mean {curve}']), max_val)
            min_val = min(np.min(multiplier * df[f'Mean {curve}']), min_val)
        else:
            breakdowns = df[breakdown_column].unique()
            for breakdown in breakdowns:
                subset_df = df[df[breakdown_column] == breakdown]
                plt.plot(subset_df['Iteration'], multiplier * subset_df[f'Mean {curve}'],
                         label=f'{breakdown_column}: {breakdown}')
                plt.fill_between(subset_df['Iteration'],
                                 multiplier * subset_df[f'Mean {curve}'] - subset_df[f'Std {curve}'],
                                 multiplier * subset_df[f'Mean {curve}'] + subset_df[f'Std {curve}'], alpha=alpha)
                max_val = max(np.max(multiplier * df[f'Mean {curve}']), max_val)
                min_val = min(np.min(multiplier * df[f'Mean {curve}']), min_val)

        title = f'{curve} vs. Iterations'
        if title_details is not None:
            title = title + f', {title_details}'

        plt.xlabel('Iterations')
        plt.ylabel(curve)
        plt.title(title)
        # Add vertical dotted lines
        plt.axhline(max_val, linestyle='dotted')
        plt.axhline(min_val, linestyle='dotted')
        plt.yticks(list(plt.yticks()[0]) + [min_val, max_val])
        plt.legend()
        plt.grid(True)

        os.makedirs(Path(file_name).parent.absolute(), exist_ok=True)
        plt.savefig(file_name)
        plt.clf()
