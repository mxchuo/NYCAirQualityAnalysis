import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['Start_Date'] = pd.to_datetime(df['Start_Date'])
    return df


#############################################################

# H1


def ozone_comparisons(df):
    # Filter rows where Measure Info contains "mean"
    mean_rows = df[df['Measure Info'].str.contains('Mean', na=False)]
    filtered_rows = mean_rows[mean_rows['Geo Place Name'].isin(['Hunts Point - Mott Haven', 'Fordham - Bronx Pk'])]
    final_filtered_rows = filtered_rows[
        filtered_rows['Name'].isin(['Nitrogen dioxide (NO2)', 'Fine particles (PM 2.5)', 'Ozone (O3)'])]

    required_rows = df[df['Geo Place Name'].isin(['Hunts Point - Mott Haven', 'Fordham - Bronx Pk']) &
                       df['Name'].isin(['Nitrogen dioxide (NO2)', 'Fine particles (PM 2.5)', 'Ozone (O3)'])]

    output_csv_path = "required_data.csv"

    required_rows.to_csv(output_csv_path, index=False)

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Iterate over the unique pollutants
    for pollutant in required_rows['Name'].unique():
        # Filter the data for the current pollutant
        pollutant_data = required_rows[required_rows['Name'] == pollutant]
        # Plot the data
        plt.bar(pollutant_data['Geo Place Name'], pollutant_data['Data Value'], label=pollutant)

    # Add labels and legend
    plt.xlabel('Location')
    plt.ylabel('Data Value')
    plt.title('Air Quality Comparison')
    plt.xticks(rotation=45)
    plt.legend(title='Pollutant')
    plt.tight_layout()

    # Show the plot
    plt.show()


def no2_comparisons(df):
    # Filter rows where Measure Info contains "mean"
    mean_rows = df[df['Measure Info'].str.contains('Mean', na=False)]
    filtered_rows = mean_rows[mean_rows['Geo Place Name'].isin(['Hunts Point - Mott Haven', 'Fordham - Bronx Pk'])]
    final_filtered_rows = filtered_rows[
        filtered_rows['Name'].isin(['Nitrogen dioxide (NO2)', 'Fine particles (PM 2.5)', 'Ozone (O3)'])]

    required_rows = df[df['Geo Place Name'].isin(['Hunts Point - Mott Haven', 'Fordham - Bronx Pk']) &
                       df['Name'].isin(['Nitrogen dioxide (NO2)'])]

    output_csv_path = "required_data.csv"

    required_rows.to_csv(output_csv_path, index=False)

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Iterate over the unique pollutants
    for pollutant in required_rows['Name'].unique():
        # Filter the data for the current pollutant
        pollutant_data = required_rows[required_rows['Name'] == pollutant]
        # Plot the data
        plt.bar(pollutant_data['Geo Place Name'], pollutant_data['Data Value'], label=pollutant)

    # Add labels and legend
    plt.xlabel('Location')
    plt.ylabel('Data Value')
    plt.title('Air Quality Comparison')
    plt.xticks(rotation=45)
    plt.legend(title='Pollutant')
    plt.tight_layout()

    # Show the plot
    plt.show()


def fp_comparisons(df):
    # Filter rows where Measure Info contains "mean"
    mean_rows = df[df['Measure Info'].str.contains('Mean', na=False)]
    filtered_rows = mean_rows[mean_rows['Geo Place Name'].isin(['Hunts Point - Mott Haven', 'Fordham - Bronx Pk'])]
    final_filtered_rows = filtered_rows[
        filtered_rows['Name'].isin(['Nitrogen dioxide (NO2)', 'Fine particles (PM 2.5)', 'Ozone (O3)'])]

    required_rows = df[df['Geo Place Name'].isin(['Hunts Point - Mott Haven', 'Fordham - Bronx Pk']) &
                       df['Name'].isin(['Fine particles (PM 2.5)'])]

    output_csv_path = "required_data.csv"

    required_rows.to_csv(output_csv_path, index=False)

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Iterate over the unique pollutants
    for pollutant in required_rows['Name'].unique():
        # Filter the data for the current pollutant
        pollutant_data = required_rows[required_rows['Name'] == pollutant]
        # Plot the data
        plt.bar(pollutant_data['Geo Place Name'], pollutant_data['Data Value'], label=pollutant, color='orange')

    # Add labels and legend
    plt.xlabel('Location')
    plt.ylabel('Data Value')
    plt.title('Air Quality Comparison')
    plt.xticks(rotation=45)
    plt.legend(title='Pollutant')
    plt.tight_layout()

    # Show the plot
    plt.show()


#############################################################

# H2

def plot_pollutant_trends(df):
    pollutants = df['Name'].unique()
    plt.figure(figsize=(15, 10))
    for pollutant in pollutants:
        subset = df[df['Name'] == pollutant]
        subset = subset.groupby(subset['Start_Date'].dt.year).agg({'Data Value': 'mean'}).reset_index()
        plt.plot(subset['Start_Date'], subset['Data Value'], label=pollutant)
    plt.title('Trends Over Time for Different Pollutants')
    plt.xlabel('Year')
    plt.ylabel('Average Concentration')
    plt.legend()
    plt.show()


###############################################################

# H3

def compute_and_plot_statistics(df, pollutant, city):
    df['Start_Date'] = pd.to_datetime(df['Start_Date'])

    grouped_df = df.groupby(['Name', 'Geo Place Name', df['Start_Date'].dt.year])

    mean_levels = grouped_df['Data Value'].mean().reset_index(name='Mean Value')
    median_levels = grouped_df['Data Value'].median().reset_index(name='Median Value')
    merged_stats = pd.merge(mean_levels, median_levels, on=['Name', 'Geo Place Name', 'Start_Date'])

    data_to_plot = merged_stats[(merged_stats['Name'] == pollutant) &
                                (merged_stats['Geo Place Name'] == city)]

    if data_to_plot.empty:
        print(f"No data available for {pollutant} in {city}.")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(data_to_plot['Start_Date'], data_to_plot['Mean Value'], label='Mean', marker='o', linestyle='-')
    ax.plot(data_to_plot['Start_Date'], data_to_plot['Median Value'], label='Median', marker='x', linestyle='--')

    ax.set_title(f'Annual Mean and Median Levels of {pollutant} in {city}')
    ax.set_xlabel('Year')
    ax.set_ylabel('Pollutant Level')
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_no2_pm25_comparison_top_cities(df, period, top_n=10):
    # Filter the DataFrame for NO2 and PM 2.5 levels for the specified time period
    df_filtered = df[(df['Time Period'].str.contains(period)) &
                     ((df['Name'] == 'Nitrogen dioxide (NO2)') |
                      (df['Name'] == 'Fine particles (PM 2.5)'))]

    pivot_df = df_filtered.pivot_table(values='Data Value',
                                       index='Geo Place Name',
                                       columns='Name',
                                       aggfunc='mean').dropna()

    top_cities = pivot_df.sort_values(by='Nitrogen dioxide (NO2)', ascending=False).head(top_n)

    # Plot the bar graph for the top N cities
    top_cities.plot(kind='bar', figsize=(15, 8))
    plt.title(f'Comparative Levels of NO2 and PM 2.5 in {period}')
    plt.ylabel('Mean Pollution Level')
    plt.xlabel('Geographical Location')
    plt.xticks(rotation=45)
    plt.legend(title='Pollutant')
    plt.tight_layout()
    plt.show()


def plot_regression_model(df, pollutant):
    df = df[(df['Name'] == pollutant)]
    x = df['Start_Date'].dt.year.to_numpy()
    y = df['Data Value'].to_numpy()

    m, b = np.polyfit(x, y, 1)
    yfit = np.polyval([m, b], x)

    plt.plot(x, yfit)
    plt.scatter(x, y)
    plt.title(f'Regression Model of {pollutant}')
    plt.xlabel('Year')
    plt.ylabel('Pollutant Level')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    predicted_value = max(0, m * 2030 + b)
    print(f'Predicted {pollutant} Level in 2030: {predicted_value}')
    df = df[(df['Start_Date'].dt.year == 2013)]
    original_mean = df['Data Value'].mean()
    print(
        f'{predicted_value} is {((original_mean - predicted_value) / original_mean) * 100}% less than the {pollutant} level in 2013 of {original_mean}')

def seasonal_statistical_comparison(df, city):
    pm25_data = df[
        (df['Name'].str.contains('Fine particles (PM 2.5)', regex=False)) & (df['Geo Place Name'] == city)].copy()
    pm25_data['Year'] = pm25_data['Start_Date'].dt.year
    grouped_data = pm25_data.groupby(['Year', 'Time Period'])

    mean_levels = grouped_data['Data Value'].mean().unstack()
    median_levels = grouped_data['Data Value'].median().unstack()

    fig, ax = plt.subplots(2, 1, figsize=(17, 17))
    for period in mean_levels.columns:
        if 'Annual Average' in period:
            continue
        label = f'{period} Mean'
        marker = 'o' if 'Summer' in period else '^'
        linestyle = '-' if 'Summer' in period else '--'
        ax[0].plot(mean_levels.index, mean_levels[period], label=label, marker=marker, markersize=8, linestyle=linestyle)

        label = f'{period} Median'
        marker = 'o' if 'Summer' in period else '^'
        linestyle = '-' if 'Summer' in period else '--'
        ax[1].plot(median_levels.index, median_levels[period], label=label, marker=marker, markersize=8, linestyle=linestyle)

    ax[0].set_title(f'Mean Levels of PM 2.5 in {city}')
    ax[1].set_title(f'Median Levels of PM 2.5 in {city}')
    ax[0].set_ylabel('PM 2.5 Mean Levels (mcg/m3)')
    ax[1].set_ylabel('PM 2.5 Median Levels (mcg/m3)')
    ax[0].set_xlabel('Year')
    ax[1].set_xlabel('Year')
    ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1), title='Time Period', fontsize='small', markerscale=1.3,
                 ncol=2)
    ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1), title='Time Period', fontsize='small', markerscale=1.3,
                 ncol=2)
    ax[0].grid(True)
    ax[1].grid(True)
    plt.tight_layout(pad=6.0)
    plt.show()