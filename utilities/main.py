def main():
    import sys
    sys.path.append('/scripts')
    from scripts import setup
    args = argsetup()
    parser = argparse.ArgumentParser(description='Sirius Data Processing Pipeline')
    args, unknown = parser.parse_known_args()


if __name__ == '__main__':
    import cProfile
    cProfile.run('main()')
    # main()