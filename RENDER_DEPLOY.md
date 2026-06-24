# Deployment Guide: Render.com (Docker Web Service)

This guide walks you through deploying the Lung Cancer Detection AI Dashboard to Render using the optimized Docker container layout.

## Prerequisites
1. A **GitHub repository** containing the project files.
2. A **Render account** (connects directly to GitHub).
3. A **MongoDB database** (e.g., MongoDB Atlas free tier cluster).

---

## Step 1: Set up MongoDB Atlas (Database Cloud)
1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) and log in.
2. Create a free shared cluster.
3. Under **Database Access**, create a user with read/write privileges.
4. Under **Network Access**, temporarily add `0.0.0.0/0` (allow access from anywhere) since Render outbound IPs change. Alternatively, use a static outbound IP service or whitelisting proxy.
5. Retrieve your connection string from the **Connect** wizard. It will look like:
   `mongodb+srv://<username>:<password>@cluster0.xxxx.mongodb.net/?retryWrites=true&w=majority`

---

## Step 2: Deploy to Render
1. Log in to the [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** and select **Web Service**.
3. Select **Connect a repository** and select your lung cancer detection repository.
4. Configure the Web Service settings:
   - **Name**: `lung-cancer-detection-ai` (or any unique name).
   - **Region**: Select a region close to your target users (e.g., Oregon, Frankfurt).
   - **Branch**: `main` (or whichever branch you want to deploy).
   - **Runtime**: **Docker** (Render will automatically detect the `Dockerfile` at the root and build it).
   - **Instance Type**: Select **Free** (or Starter for higher CPU/RAM constraints).

---

## Step 3: Configure Environment Variables on Render
Scroll down to the **Advanced** section or go to the **Env Groups** / **Environment** tab:
Add the following environment variables:
1. `MONGO_URI` = `mongodb+srv://<username>:<password>@cluster0.xxxx.mongodb.net/?retryWrites=true&w=majority` (Substitute with your Atlas connection string).
2. `MONGO_DB_NAME` = `lung_cancer_db`
3. `STREAMLIT_SERVER_PORT` = `8501`
4. `PORT` = `8501`

---

## Step 4: Build and Deploy
1. Click **Create Web Service**.
2. Render will trigger a build from the Docker image.
   *   *Note: Our `Dockerfile` automatically installs the CPU-only version of PyTorch, which is light and highly memory efficient. This prevents the build from exceeding Render's memory limits and crashing during setup.*
3. Once the build completes and logs output `Network URL: http://0.0.0.0:8501`, the status will change to **Live**.
4. Open the generated sub-domain link (e.g. `https://lung-cancer-detection-ai.onrender.com`) to access your application.

---

## Troubleshooting & Tips
- **Cold Starts**: Render's Free tier spins down web services after 15 minutes of inactivity. When you visit the app after it spins down, it can take 1-2 minutes to restart.
- **SQLite Database**: If the MongoDB connection fails or is misconfigured, the application will fallback to the SQLite backup at `data/history.db`. However, note that Render's file system is ephemeral. Any data saved to SQLite will be wiped out when the service restarts. Connecting to MongoDB is highly recommended for production history logs.
