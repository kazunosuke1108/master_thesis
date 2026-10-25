FROM node:24-bookworm-slim AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY tsconfig.json ./
COPY src ./src
RUN npm run build

FROM node:24-bookworm-slim
ARG CODEX_VERSION=0.163.0
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupmod -g ${GROUP_ID} node && usermod -u ${USER_ID} -g ${GROUP_ID} node
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates poppler-utils python3 && rm -rf /var/lib/apt/lists/*
RUN npm install --global @openai/codex@${CODEX_VERSION}
WORKDIR /app
COPY --from=build /app/dist ./dist
COPY package.json ./
RUN mkdir -p /data /codex /workspace && chown node:node /data /codex /workspace
USER node
ENV HOST=0.0.0.0 PORT=8787 DATA_DIR=/data CODEX_HOME=/codex RESEARCH_WORKSPACE=/workspace
EXPOSE 8787
CMD ["node", "dist/server.js"]
