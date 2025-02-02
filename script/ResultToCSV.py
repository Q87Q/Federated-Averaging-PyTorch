import csv


def CreateHeader():
    header = ["", "Dataset", "Model", "DataType", "Number of Client", "Number of Epoch", "Algorithm Execution Time",
              "Accuracy"]

    f = open('./result/result.csv', 'w+', encoding='UTF8', newline='')

    writer = csv.writer(f)

    writer.writerow(header)

    f.close()


def CreateResultData(Case, Dataset, Model, DataType, Number_of_Client, Number_of_Epoch, Accuracy, Total_time):
    data = [Case, Dataset, Model, DataType, Number_of_Client, Number_of_Epoch, Total_time, Accuracy]

    f = open('./result/result.csv', 'a', encoding='UTF8', newline='')

    writer = csv.writer(f)

    writer.writerow(data)

    f.close()

def Save_Accuracy_of_each_epoch(mode, casename, Accuracy_list, Best_Accuracy):
    filename = ""

    filepath = "./result/Accuracy_of_each_epoch.csv"

    if mode == 0:
        f = open(filepath, 'w+', encoding='UTF8', newline='')

    elif mode == 1:
        f = open(filepath, 'a', encoding='UTF8', newline='')

    writer = csv.writer(f)
    if mode == 0:
        header = [""]
        for i in range(len(Accuracy_list)):
            string = "round " + str(i + 1)
            header.append(string)
        header.append("")
        header.append("Best Accuracy")
        writer.writerow(header)

    data = [casename]
    for i in range(len(Accuracy_list)):
        data.append(Accuracy_list[i])
    data.append("")
    data.append(Best_Accuracy)
    writer.writerow(data)

    f.close()
