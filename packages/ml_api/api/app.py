from flask import Flask

from api.config import get_logger

#additional packages for flasgger endpoint
from flasgger import Swagger
#end of additional packages

_logger = get_logger(logger_name=__name__)


def create_app(*,config_object) -> Flask:
    """
    Create a flask app instance.
    """

    flask_app = Flask('ml_api')
    flask_app.config.from_object(config_object)

   # import blueprints
    from api.controller import prediction_app
    flask_app.register_blueprint(prediction_app)
    _logger.debug('Application instance created')

    return flask_app

###def create_app_for_Flasgger(*,config_object) -> Flask:
def create_app_for_Flasgger() -> Flask:
    """
    Create a flask app instance for Flasger.
    """

    flask_app_flasgger = Flask(__name__)
    ###flask_app_flasgger.config.from_object(config_object)

    swagger = Swagger(flask_app_flasgger)
    return (flask_app_flasgger,swagger)
