import argparse
import preprocessors.libritts as libritts

def main(data_path, out_dir, sr):
    libritts.prepare_align_and_resample(data_path, out_dir, sr)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/hcy71/DATA/LibriTTS')
    parser.add_argument('--out_dir', type=str, default='/home/hcy71/DATA/preprocessed_data/LibriTTS')
    parser.add_argument('--resample_rate', '-sr', type=int, default=22050)

    args = parser.parse_args()

    main(args.data_path, args.out_dir, args.resample_rate)