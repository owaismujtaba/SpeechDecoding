from src.components.data.feature_extractor import MFCC_Extractor


if __name__ == "__main__":
    for sub in range(1, 31):
        subject_id = str(sub).zfill(2)
        print(f"Processing subject {subject_id}")

        extractor = MFCC_Extractor(subject=subject_id)
        extractor.read_audio()
        extractor.get_mfcc_world_vocoder()
        extractor.save_features()
