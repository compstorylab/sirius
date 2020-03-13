# Sirius: Exploratory Analysis Tool
We aim to build an exploratory analysis tool for data scientists to find meaningful relationships between variables. The
tool will feature a web-based dashboard showing a network graph of variables in a high-dimensional data set, with edges 
drawn according to a mutual information statistic that is computed server-side. Users can hover over variables to see 
related variables in a given cluster, and can click on edges between variables to see a graph showing a comparison between 
variable pairs.

The purpose of this tool is to aid data scientists in understanding information gain across a large swath of variables 
through meta-analysis, when visually assessing each possible permutation of pairwise variable combination charts would 
be a daunting task, given the size of the data set. This analytic tool could therefore inform data scientists in choosing 
features for imputation models, as well as indicate potential unexpected relationships between variables

## Getting Started
We will walk through how to set up this tool in your local environment. If you would like to participate in the development,
welcome to create a pull request.
### Prerequisites
`python 3` and `git` are required before you start the setting up process
### Setting Up the Tool
1. Clone the repo to your local directory
2. Enter the directory, create a virtual environment and activate it

    ```
    python3 -m venv venv
    source venv/bin/activate
    
    ```
    
3. Install required packages

    ```pip install -r requirements.txt```
4. Create a .env file under the project folder, which contains 
    ```text
       SIRIUS_SETTINGS_SECRET_KEY={your string value here with quotes}
       (this one is optional)ENVIRONMENT={"dev" or "qa" or "prod"}
    ```
    secret key should be set to a unique, unpredictable value. It is used for sessions, messages, PasswordResetView tokens
    or any usage of is cryptographic signing unless a different key is provided.
5. Execute the following command to set up the database structure

    ```python manage.py migrate```    
6. Start the server

    ```python manage.py runserver```
    
Congratulations, the Exploratory Analysis Tool is live in your local environment!
You can access it by the url returned from the above command, usually it is [http://127.0.0.1:8000](http://127.0.0.1:8000)

## User Guide of the Tool

1. Upload the json file with structures like:
```json
{"nodes": [
    {"name": "string_name_1",
    "type": "continuous OR discrete",
    "neighbors": ["string_name_1", "string_name_2", "string_name_3"]}],
 
"links": [
    {"source": "string_name_of_source_node",
    "target": "string_name_of_target_node",
    "weight": 0.8986,
    "viztype": "CC or DC or DD"}]}

```
There is an example json under `utilities/example_data`

2. You should see a network graph with which to explore feature pairs with high mutual information.

<b>Note<b>: Please use "_" as delimiters between features/variables name, for example, "feature1_feature2"


## Data Processing

Sirius works by processing cleaned data to compute pairwise feature relationships using mutual information, and generating a network graph layout for exploratory analysis. The following diagram may be helpful for understanding:

![Sirius Data Processing Flowchart](https://raw.githubusercontent.com/compstorylab/sirius/develop/static/documentation/flowchart.png)

All data processing takes place in `matrix.py`. There are a number of customizable parameters in this script:

Argument | Type | Default | Description
------------ | ------------ | ------------- | ------------ | -------------
`--dpi` | int | `150` | Resolution of output plots
`--discrete-threshold` | int | `5` | Number of responses below which numeric features are considered discrete
`--chart` | boolean | `False` | Display images while running computation
`--charter` | choice | `'Plotly'` | The plotting library to use. Options: `'Plotly'` or `'Seaborn'`
`--debug` | boolean | `False` | Print updates to the console while running
`--output` | boolean | `False` | Output json and pngs to files
`--no-viz` | boolean | `False` | Do not output pair plots, network graph image, or chart json
`--no-mi` | boolean | `False` | Do not compute MI. Use cached MI values instead
`--cache` | boolean | `False` | Cache MI values to use later when generating visualizations
`--sample-n` | int | `None` | Subsample the data. By default, work with all the data
`--input-file` | string | `'example_data/data.csv'` | Location of the input data
`--output-dir` | string | `'example_data/output'` | A directory in which to store the output json and png files



## Development
If you would like to contribute to the development, WELCOME!

Please make sure you have `npm` installed.
Then `npm install` to install all required js libraries for development.
We use typescript in this project. Please execute `./node_modules/.bin/webpack` to compile typescript into javascript.

Thorough documentation is required in development.



