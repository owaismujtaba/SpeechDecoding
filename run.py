
from src.components.data.eeg_mfcc_loader import DataLoader
from src.components.model.trainer import Trainer
from src.components.model.neuroincept_decoder import NeuroInceptDecoder

import pdb

if __name__ == "__main__":
    for sub in range(1, 31):
        subject_id = str(sub).zfill(2)
        print(f"Processing subject {subject_id}")

        dataloader =  DataLoader(subject_id=subject_id) #MFCC_Extractor(subject=subject_id)
        eeg, mfcc = dataloader.align_stack_eeg_with_mfcc()

        #eeg = eeg[:3000]
        #mfcc = mfcc[:3000]
        print(f'sub-{subject_id}, {eeg.shape, mfcc.shape}')
        
        trainer = Trainer(
            subject=subject_id,
            mfcc=mfcc, eeg=eeg
        )

        model = NeuroInceptDecoder(
            input_shape=(eeg.shape[1],), output_shape=50
        )

        trainer.train(model=model, model_name='neuroincept')

        
