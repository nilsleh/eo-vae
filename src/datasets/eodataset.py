import dataset4eo as eodata
import litdata as ld

DATA_DIR = "optimized_flair2_test"

def get_flair_dataloader(data_dir=DATA_DIR, batch_size=1):
    Flair_dataset = eodata.StreamingDataset(input_dir=DATA_DIR,num_channels=5,channels_to_select=[0,1,2])
    dataloader = ld.StreamingDataLoader(Flair_dataset, batch_size=batch_size, drop_last=False)
    return dataloader
