from LeafSegmentorDownload import download
from LeafSegmentorTrain import train
from LeafSegmentorInfer import infer
from Reference import HelpReference
from LeafSegmentorInfo import info
from LeafSegmentorCut import cut
import argparse


def main():
    # top level parser
    parser = argparse.ArgumentParser(description=HelpReference.description, add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help=HelpReference.help_description)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
    subparsers = parser.add_subparsers()

    # parser for download
    parser_download = subparsers.add_parser('download', help=HelpReference.DownloadReference.description)
    parser_download.add_argument('task_id', help=HelpReference.DownloadReference.task_id)
    parser_download.add_argument('location', nargs='?', help=HelpReference.DownloadReference.location, default='downloads')
    parser_download.set_defaults(func=download)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
