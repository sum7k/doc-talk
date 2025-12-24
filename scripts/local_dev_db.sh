#!/bin/bash
set -e

# Local development Postgres database container management
# Usage: ./scripts/local_dev_db.sh [start|stop|restart|logs|destroy]

CONTAINER_NAME="fastapi-skeleton-postgres"
POSTGRES_USER="postgres"
POSTGRES_PASSWORD="postgres"
POSTGRES_DB="fa_skeleton_dev"
POSTGRES_PORT="5432"

case "$1" in
  start)
    echo "🚀 Starting Postgres container..."
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
      echo "✅ Container is already running"
    elif [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
      echo "▶️  Starting existing container..."
      docker start $CONTAINER_NAME
    else
      echo "📦 Creating new Postgres container..."
      docker run -d \
        --name $CONTAINER_NAME \
        -e POSTGRES_USER=$POSTGRES_USER \
        -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
        -e POSTGRES_DB=$POSTGRES_DB \
        -p $POSTGRES_PORT:5432 \
        -v fastapi-skeleton-pgdata:/var/lib/postgresql/data \
        postgres:16-alpine

      echo "⏳ Waiting for Postgres to be ready..."
      sleep 3
    fi

    echo ""
    echo "✅ Postgres is running!"
    echo "📝 Connection string:"
    echo "   postgresql+asyncpg://$POSTGRES_USER:$POSTGRES_PASSWORD@localhost:$POSTGRES_PORT/$POSTGRES_DB"
    echo ""
    echo "💡 Run migrations with: make migrate"
    ;;

  stop)
    echo "⏹️  Stopping Postgres container..."
    docker stop $CONTAINER_NAME 2>/dev/null || echo "Container not running"
    echo "✅ Container stopped"
    ;;

  restart)
    echo "🔄 Restarting Postgres container..."
    docker restart $CONTAINER_NAME
    echo "✅ Container restarted"
    ;;

  logs)
    echo "📋 Showing Postgres logs (Ctrl+C to exit)..."
    docker logs -f $CONTAINER_NAME
    ;;

  destroy)
    echo "🗑️  Destroying Postgres container and data..."
    read -p "⚠️  This will delete ALL data. Continue? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      docker stop $CONTAINER_NAME 2>/dev/null || true
      docker rm $CONTAINER_NAME 2>/dev/null || true
      docker volume rm fastapi-skeleton-pgdata 2>/dev/null || true
      echo "✅ Container and data destroyed"
    else
      echo "❌ Cancelled"
    fi
    ;;

  status)
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
      echo "✅ Container is running"
      docker ps --filter name=$CONTAINER_NAME --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    elif [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
      echo "⏸️  Container exists but is stopped"
    else
      echo "❌ Container does not exist"
    fi
    ;;

  *)
    echo "Usage: $0 {start|stop|restart|logs|destroy|status}"
    echo ""
    echo "Commands:"
    echo "  start    - Start Postgres container (creates if needed)"
    echo "  stop     - Stop the container"
    echo "  restart  - Restart the container"
    echo "  logs     - Show container logs"
    echo "  destroy  - Remove container and all data"
    echo "  status   - Check container status"
    exit 1
    ;;
esac
