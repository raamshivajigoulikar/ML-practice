import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import os
os.chdir("C:\\Users\\rgoulika\\Downloads\\Datasets\\Datasets")

student_info=pd.read_csv("studentInfo.csv")

sns.countplot(x="final_result",hue="code_module",data=student_info)

sns.countplot(x="final_result",hue="region",data=student_info)

sns.countplot(x="age_band",hue="final_result",data=student_info)

sns.countplot(y="final_result",hue="gender",data=student_info)

sns.countplot(y="gender",hue="final_result",data=student_info)

sns.countplot(y="gender",data=student_info)

sns.countplot(y="final_result",hue="highest_education",data=student_info)

sns.countplot(y="final_result",hue="disability",data=student_info)

