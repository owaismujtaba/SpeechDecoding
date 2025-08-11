
from src.components.data.eeg_mfcc_loader import DataLoader
from src.components.model.trainer import Trainer
from src.components.model.neuroincept_decoder import NeuroInceptDecoder

from src.components.data.audio_maker import AudioMaker
import config as config
import pdb

if __name__ == "__main__":
    for sub in range(1, 31):
        subject_id = str(sub).zfill(2)
        print(f"Processing subject {subject_id}")

        if config.train:

            dataloader =  DataLoader(subject_id=subject_id) #MFCC_Extractor(subject=subject_id)
            eeg, mfcc = dataloader.align_stack_eeg_with_mfcc()

            print(f'sub-{subject_id}, {eeg.shape, mfcc.shape}')
            
            trainer = Trainer(
                subject=subject_id,
                mfcc=mfcc, eeg=eeg
            )

            model = NeuroInceptDecoder(
                input_shape=(eeg.shape[1],), output_shape=50
            )

            trainer.train(model=model, model_name='neuroincept')

        if config.audio_reconstruction:

            audio_maker = AudioMaker(
                subject=subject_id,
                model='neuroincept'
            )

            audio_maker.create_audio()

        pdb.set_trace()