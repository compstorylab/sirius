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
5. Execute the following command to set up the database structure

    ```python manage.py migrate```    
6. Start the server

    ```python manage.py runserver```
    
Congratulations, the Exploratory Analysis Tool is live in your local environment!
You can access it by the url returned from the above command, usually it is [http://127.0.0.1:8000](http://127.0.0.1:8000)

## User Guide of the Tool
1. upload the json file with structures like:
```json
{
"nodes": [{"index":1, "source": "node_1"}, {"index":2, "source": "node_2"}]
,
"links":[{"source": 1, "target": 2, "correlations": 0.58, "link": "node_1_to_2", "support": 300}]
}
```
(the json file can be generated from xxx (will be updated))
2. viola! you will see a network graph with options to explore with.

<b>Note<b>:
1. please use "_" as delimiters between features/variables name

## Development
If you would like to contribute to the development, WELCOME!

Please make sure you have `npm` installed.
We use typescript in this project. Please execute `webpack` to compiles typescript into javascript.

Thorough docstring is required in development



