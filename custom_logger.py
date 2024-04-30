import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='../jobapp.log',
                    filemode='w')

logger = logging.getLogger(__name__)