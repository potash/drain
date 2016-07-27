import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=0)

import drain.yaml
drain.yaml.configure()
