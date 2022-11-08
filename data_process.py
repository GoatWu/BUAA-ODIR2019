import logging
import logging.config
import time
from data_utils.data_parser import DataParser


def main(file, generate_path):
    # Produce all the patient information and store it in an dictionary
    start = time.time()
    logger.debug('Produce all the patient information and store it in an dictionary')
    parser = DataParser(file, 'Sheet1')
    logger.debug('File ' + file + ' parsed successfully!')
    logger.debug('Excel to Patients DTO')
    patients = parser.generate_person()
    logger.debug('Excel to Patients DTO Finished')

    # Additional quality check, can be commented out
    logger.debug('Ensuring Training Data Quality')
    parser.check_data()
    logger.debug('Ensuring Training Data Quality Finished')

    # Additional CSV for single class labelling
    logger.debug('Ensuring CSV generation')
    parser.generate_csv(generate_path)
    logger.debug('Ensuring CSV generation Finished')

    end = time.time()
    logger.debug('All Done in ' + str(end - start) + ' seconds')


if __name__ == '__main__':
    # create logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('Main')
    file_dir = '/mnt/d/MyDataBase/ODIR-5K/ODIR-5K_Training_Annotations(Updated)_V2.xlsx'
    generate_file = './data_result.csv'
    main(file_dir, generate_file)
