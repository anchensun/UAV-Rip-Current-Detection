if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Super Resolution OffShore Back to Normal')
    parser.add_argument('oirPath', type=str, help='oirPath')
    parser.add_argument('companyName', type=str, help='companyName')
    parser.add_argument('companyType', type=str, help='companyType')

    args = parser.parse_args()

    oirPath = args.oirPath
    companyName = args.companyName
    companyType = args.companyType

