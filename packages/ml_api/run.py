from api.app import create_app
from api.config import DevelopmentConfig



application = create_app(
    config_object=DevelopmentConfig)

######################
#from api.app import create_app_for_Flasgger
# application_flasgger = create_app_for_Flasgger()
from api import controller
application_flasgger = controller.prediction_app_swagger
######################

if __name__ == '__main__':
    #application.run()
    application_flasgger.run()

