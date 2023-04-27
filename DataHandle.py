import pandas as pd
import numpy as np
import json
data = pd.read_csv("/Users/dhr/PycharmProjects/chatglm-finetune/content/train.csv",header=None)
data.columns = ["question","context","dsl"]

for idx,dataItem in data.iterrows():
    if idx == 0:
        continue
    question = dataItem["question"]
    context = dataItem["context"]
    dsl = dataItem["dsl"]

    # 处理input
    contextData = pd.read_json(context)
    header = ""
    for idx, headerItem in contextData["header"].items():
        if idx == 0:
            header += "`" + str(headerItem) + " text"
        header += "`,\n`" + str(headerItem) + "` text"
    tableSchema = "table schema：{\n" + header + "\n}"
    input = tableSchema+"\n"+question;

    # 处理output
    dslData = json.loads(dsl)
    output = f"SELECT {dslData['select']['column']} FROM {dslData['from']} WHERE {dslData['where']}";
    print(output)

