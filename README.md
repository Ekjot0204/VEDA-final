# VEDA GitHub Pages frontend

Publish the contents of this folder to the root of the GitHub Pages repository. The frontend remains static; AI processing and private image storage require the separately deployed `veda-backend` service.

## Configure backend + cloud image storage
1. Deploy the `veda-backend` service and configure the private Supabase Storage bucket using `veda-backend/README.md`.
2. In VEDA open **More → Privacy & security**.
3. Enter the deployed backend URL under **AI backend connection** and save.
4. Enter the Supabase Project URL and anon/publishable key under **Private cloud image storage** and save.
5. Create a VEDA cloud account or sign in.
6. Scan an image/PDF and choose **Save to my records**. The original file is uploaded to private cloud storage; local health-record metadata continues to be stored in this browser.

Never put the Supabase service-role key or OpenAI API key in the frontend or GitHub repository. The cloud storage setup is a prototype and should not be used with real patient records until privacy/security controls have been reviewed.
