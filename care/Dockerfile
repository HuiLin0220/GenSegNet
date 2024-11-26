FROM python:3.9

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN adduser --system --group user
USER user

WORKDIR /workspace

COPY --chown=user:user requirements.txt .
COPY --chown=user:user inference.py .
COPY --chown=user:user model_folder_fine ./model_folder_fine
COPY --chown=user:user model_folder_coarse ./model_folder_coarse
COPY --chown=user:user nnunetv2 ./nnunetv2
COPY --chown=user:user reverse.py .
COPY --chown=user:user crop.py .

RUN pip3 install --user --no-cache-dir -r ./requirements.txt


ENTRYPOINT ["python", "inference.py"]