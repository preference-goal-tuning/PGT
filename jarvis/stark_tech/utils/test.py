# Copyright (c) 2020 All Rights Reserved


import os
import logging

SHOULD_ASSERT = bool(os.environ.get('MINERL_TESTING', False)) or 'PYTEST_CURRENT_TEST' in os.environ
