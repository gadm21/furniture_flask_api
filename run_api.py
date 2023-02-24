

import apiwrapper

from apiwrapper.utils import load_trained_model

model = load_trained_model()
app = apiwrapper.Api(model=model)

application = app.app

if __name__ == "__main__":
    application.run(debug=True, host='0.0.0.0', port=3000)