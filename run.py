
from src.components.data.eeg_mfcc_loader import DataLoader


if __name__ == "__main__":
    for sub in range(1, 31):
        subject_id = str(sub).zfill(2)
        print(f"Processing subject {subject_id}")

        dataloader =  DataLoader(subject_id=subject_id) #MFCC_Extractor(subject=subject_id)
        eeg, mfcc = dataloader.align_stack_eeg_with_mfcc()
        print(f'sub-{subject_id}, {eeg.shape, mfcc.shape}')
