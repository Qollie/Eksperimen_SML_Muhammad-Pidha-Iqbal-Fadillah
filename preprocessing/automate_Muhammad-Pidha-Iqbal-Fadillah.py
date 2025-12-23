import pandas as pd
from sklearn.preprocessing import StandardScaler

def LoadDataset(filePath):
    dataset = pd.read_csv(filePath)
    return dataset


def PreprocessDataset(dataset):
    # Memisahkan fitur dan target
    FeatureDf = dataset.drop("PaymentTier", axis=1)
    TargetSeries = dataset["PaymentTier"]

    # Encoding data kategorikal
    EncodedFeatureDf = pd.get_dummies(FeatureDf, drop_first=True)

    # Standarisasi fitur
    Scaler = StandardScaler()
    ScaledFeatures = Scaler.fit_transform(EncodedFeatureDf)

    # Menggabungkan kembali data hasil preprocessing
    ProcessedDf = pd.DataFrame(
        ScaledFeatures,
        columns=EncodedFeatureDf.columns
    )
    ProcessedDf["PaymentTier"] = TargetSeries.values

    return ProcessedDf


def SaveDataset(dataset, outputPath):
    dataset.to_csv(outputPath, index=False)


if __name__ == "__main__":
    InputPath = "Employee_raw.csv"
    OutputPath = "Employee_preprocessing.csv"

    RawDataset = LoadDataset(InputPath)
    ProcessedDataset = PreprocessDataset(RawDataset)
    SaveDataset(ProcessedDataset, OutputPath)

    print("Preprocessing selesai. Dataset siap digunakan.")