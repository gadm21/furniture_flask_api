# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app



# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt


# Expose port 5000
EXPOSE 3000


ENTRYPOINT ["python"]

# Run run_api.py when the container launches
CMD ["run_api.py"]

# build this by
# docker build -t fashion-flask-api .
# run this by
# docker run -p 3000:3000 fashion-flask-api