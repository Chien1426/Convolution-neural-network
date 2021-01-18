from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.feature_selection import SelectPercentile
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Ridge


def Verification(y_test, y_pred):
    __, __, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    F1List, SenList, SpeList, PreList, AccList, TPList, TNList, FPList, FNList = [], [], [], [], [], [], [], [], []

    for j in range(0, len(thresholds), 1):
        TN = 0;
        FN = 0;
        FP = 0;
        TP = 0
        # print("此時的臨界值為:{:.4f}".format(thresholds[j]))

        for i in range(0, len(y_pred), 1):

            if (y_pred[i] < thresholds[j]):
                if (y_test[i] == 0):
                    TN = TN + 1
                else:
                    FN = FN + 1
            else:
                if (y_test[i] == 1):
                    TP = TP + 1
                else:
                    FP = FP + 1

        if ((FN + TP) != 0):
            Sen = TP / (FN + TP)
        else:
            Sen = 0

        if ((TN + FP) != 0):
            Spe = TN / (TN + FP)
        else:
            Spe = 0

        if ((TP + FP) != 0):
            Pre = TP / (TP + FP)
        else:
            Pre = 0

        if ((Pre + Sen) != 0):
            F1 = 2 * Pre * Sen / (Pre + Sen)
        else:
            F1 = 0

        Acc = (TN + TP) / (TN + TP + FP + FN)

        F1List.append(F1)
        SenList.append(Sen)
        SpeList.append(Spe)
        PreList.append(Pre)
        AccList.append(Acc)

        TNList.append(TN)
        TPList.append(TP)
        FPList.append(FP)
        FNList.append(FN)

    BestF1 = max(F1List)

    Index = F1List.index(max(F1List))
    print("\n臨界值:{:.3f}".format(thresholds[Index]))
    print("\n準確度:{:.3f}".format(AccList[Index]))
    print("\n敏感度:{:.3f}".format(SenList[Index]))
    print("\n特異度:{:.3f}".format(SpeList[Index]))
    print("\n查準率:{:.3f}".format(PreList[Index]))

    print("\nF1分數:{:.3f}".format(max(F1List)))
    Confusion = [[TNList[Index], FPList[Index]], [FNList[Index], TPList[Index]]]
    Confusion = np.array(Confusion)
    print("\n混淆矩陣:\n{}".format(Confusion))

    return


DataFile1 = r"_.csv"
data_name1 = pd.read_csv(DataFile1)

x1 = []
y1 = []

for i1 in range(len(data_name1)):
    if((data_name1['true1'][i1] == 1) and (data_name1['true0'][i1] == 0)):
        b1 = data_name1['test1'][i1]
        x1.append(1)
        y1.append(b1)
    if((data_name1['true1'][i1] == 0) and (data_name1['true0'][i1] == 1)):
        b1 = data_name1['test1'][i1]
        x1.append(0)
        y1.append(b1)


label = "_"

fpr, tpr, thresholds = roc_curve(x1, y1)
roc_auc = auc(x=fpr, y=tpr)
plt.plot(refpr, retpr, color='blue',label='%s (AUC = %0.3f)' % (label, roc_auc))#'%s (auc = %0.3f)' % (label, roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1],[0, 1],linestyle='--',color='gray',linewidth=2,)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

print("\n∎∎∎∎∎_∎∎∎∎∎")
Verification(x1, y1)