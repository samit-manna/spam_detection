# Azure Cache for Redis
resource "azurerm_redis_cache" "main" {
  # count               = var.deploy_redis ? 1 : 0
  name                = "${var.project_name}-${var.environment}-redis-${random_string.unique.result}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  capacity            = var.redis_capacity
  family              = var.redis_family
  sku_name            = var.redis_sku_name
  minimum_tls_version = "1.2"

  tags = local.common_tags
}
