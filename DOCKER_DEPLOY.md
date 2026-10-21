# Deploy two model versions with Docker on the cloud VM

This guide runs the spam classifier web app (`app.py`) on the cloud VM at **34.220.144.141**, twice: once with model **v1** and once with model **v2**, side by side on two ports. You open both in the browser, type the same message into each, and see how the two versions disagree.

| Version | Trained on | Accuracy | Spam recall | Port |
|---|---|---|---|---|
| v1 | `data/sms_real_1k.csv` (85% ham, imbalanced) | about 86% | about 4% | 8001 |
| v2 | `data/balanced.csv` (1,000 ham + 1,000 spam) | about 99% | about 100% | 8002 |

Scores are on the fixed test set `data/test_set.csv`. v1 looks accurate but labels almost every message as ham, so it misses nearly all spam. That is the point of the test.

## What you need

- An SSH login to the VM (user name and key or password).
- Docker, `git` and `make` on the VM. You do **not** need Python on the VM: the models are trained inside a container.
- Ports **8001** and **8002** open for inbound TCP on the VM's firewall. 34.220.144.141 is an AWS address, so this is an inbound rule in the instance's **security group**.

In the commands below, replace `<user>` with your VM login and `<name>` with a short folder name, lower case, no spaces (for example `maria`). If the VM is shared, `<name>` also keeps your images and containers apart from everyone else's.

## 1. Log in to the VM

```bash
ssh <user>@34.220.144.141
```

Check that Docker works:

```bash
docker run --rm hello-world
```

If Docker, `git` or `make` is missing on an Ubuntu VM, install them, then log out and back in so the `docker` group applies:

```bash
sudo apt-get update && sudo apt-get install -y docker.io git make
sudo usermod -aG docker $USER
```

## 2. Create your folder and clone the repository

```bash
mkdir -p ~/<name>
cd ~/<name>
git clone https://github.com/iportilla/MLOps-notebooks-5350.git
cd MLOps-notebooks-5350
```

Run every command from here on inside `~/<name>/MLOps-notebooks-5350`. If you cloned it earlier, run `git pull` instead of `git clone` to get the latest code.

## 3. Create the two model versions

`make_solution_models.py` trains v1 and v2 and saves them in `models/`. This runs it in a throwaway Python container, so nothing is installed on the VM:

```bash
docker run --rm -v "$PWD":/app -w /app python:3.11-slim \
  sh -c "pip install -q -r requirements.txt && python make_solution_models.py --force && chown -R $(id -u):$(id -g) models"
```

You should see:

```
saved models/model_nb_v1.pkl and models/model_nb_v1.json: accuracy 85.6%, recall 4.0% on data/test_set.csv
saved models/model_nb_v2.pkl and models/model_nb_v2.json: accuracy 99.0%, recall 100.0% on data/test_set.csv
```

Each `.pkl` is the model and each `.json` is its model card (data used, fingerprint, metrics). Look at them with `cat models/model_nb_v1.json`.

## 4. Build one image per version

The [Dockerfile](Dockerfile) packages whichever model is at `models/production.pkl`. Copy each version there in turn and build an image tagged with the version:

```bash
cp models/model_nb_v1.pkl models/production.pkl
make build IMAGE=spam-lab-<name>:v1

cp models/model_nb_v2.pkl models/production.pkl
make build IMAGE=spam-lab-<name>:v2
```

Check that both images exist:

```bash
docker images spam-lab-<name>
```

## 5. Run both versions

```bash
make run IMAGE=spam-lab-<name>:v1 CONTAINER=spam-lab-<name>-v1 PORT=8001
make run IMAGE=spam-lab-<name>:v2 CONTAINER=spam-lab-<name>-v2 PORT=8002
```

Check that both containers are up:

```bash
docker ps --filter name=spam-lab-<name>
```

## 6. Test them in the browser

On your laptop, open both:

- v1: <http://34.220.144.141:8001>
- v2: <http://34.220.144.141:8002>

Paste each message into both apps and press **Classify**:

| Message | v1 says | v2 says |
|---|---|---|
| `URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010` | HAM (wrong) | SPAM |
| `Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's` | HAM (wrong) | SPAM |
| `Hey, are we still meeting for lunch today?` | HAM | HAM |

The model only sees two features, the message length and the number of punctuation marks (shown under **Features** in the app). Short messages look like ham to both versions, even the default `WINNER!!` text. Long messages with some punctuation are where v2 catches spam that v1 misses.

You can also test from the VM's command line without a browser:

```bash
curl -s -o /dev/null -w "v1: %{http_code}\n" http://localhost:8001
curl -s -o /dev/null -w "v2: %{http_code}\n" http://localhost:8002
```

Both should print `200`.

## 7. Promote and roll back

Each version is its own image, so switching the model behind a port is just starting a different image on it. No rebuild is needed. For example, put v1 back on port 8002 (a rollback), then return to v2:

```bash
make run IMAGE=spam-lab-<name>:v1 CONTAINER=spam-lab-<name>-v2 PORT=8002
make run IMAGE=spam-lab-<name>:v2 CONTAINER=spam-lab-<name>-v2 PORT=8002
```

`make run` stops the old container with that name before starting the new one. Refresh the browser tab after each command.

## 8. Logs and clean up

```bash
make logs CONTAINER=spam-lab-<name>-v1     # Ctrl+C to quit
make logs CONTAINER=spam-lab-<name>-v2
```

When you are done, stop both containers and delete both images:

```bash
make clean IMAGE=spam-lab-<name>:v1 CONTAINER=spam-lab-<name>-v1
make clean IMAGE=spam-lab-<name>:v2 CONTAINER=spam-lab-<name>-v2
```

## Troubleshooting

**`permission denied ... docker.sock`**

Your user is not in the `docker` group. Run `sudo usermod -aG docker $USER`, then log out and back in.

**The page at `http://34.220.144.141:8001` does not load**

Check that the container is running with `docker ps --filter name=spam-lab-<name>` and that `curl http://localhost:8001` works on the VM. If it works on the VM but not from your laptop, the port is blocked: add an inbound rule for TCP 8001 and 8002 to the instance's security group.

**`port is already allocated`**

Another container has that port. Run `docker ps --format '{{.Names}}  {{.Ports}}'` to see which, then stop it or pick other ports (for example `PORT=8003`) and open those in the security group too.

**`models/production.pkl is missing` from `make build`**

Step 4 copies a model to `models/production.pkl` before each build. Run the `cp` line first. If `models/model_nb_v1.pkl` is missing too, run step 3 again.
