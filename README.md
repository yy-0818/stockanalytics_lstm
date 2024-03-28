# stockanalytics_lstm

Show stock predictions w/ Streamlitlstm

### Clone

```
git clone https://github.com/yy-0818/stockanalytics_lstm.git
```

### Create a virtual environment

```
python -m venv venv
```

### Activate the virtual environment (Linux/macOS)

```
source venv/bin/activate
```

### Activate the virtual environment (Windows)

```
venv\Scripts\activate
```

### Install dependencies

```
python -m pip install -r requirements.txt
```

### Run the app

```
streamlit run Homepage.py
```

### View

![homepage](https://static.ivanlife.cn/imges/image-20231008220750893.png)

### Directory Structure

```
├── Stock_History_Day_K-Line
│ ├── Data
│ │ ├── combined_stock_data.csv
│ │ ├── data.ipynb
│ │ ├── stock_1.csv
│ │ ├── stock_2.csv
│ │ ├── stock_3.csv
│ │ └── stock_data_summary.csv
│ ├── Homepage.py
│ ├── Model
│ │ ├── model_1.h5
│ │ └── model_2.h5
│ ├── history.ipynb
│ ├── historyK.py
│ ├── images
│ │ ├── lstm.jpg
│ │ ├── lstm_inside.jpg
│ │ └── rnn.jpg
│ └── pages
│ ├── 📄profile_report.py
│ └── 📈Stock Price Prediction.py
├── requirements.txt
└── README.md
```
