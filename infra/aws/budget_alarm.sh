#!/usr/bin/env bash
# One-time setup: alert by email when spend crosses 75% and 90% of the $200
# credit, so a forgotten running instance can't quietly burn the whole thing.
set -euo pipefail

: "${AWS_ACCOUNT_ID:?set AWS_ACCOUNT_ID to your 12-digit account id}"
: "${ALERT_EMAIL:?set ALERT_EMAIL to where alerts should go}"
BUDGET_LIMIT="${BUDGET_LIMIT:-200}"

cat > /tmp/budget.json <<EOF
{
  "BudgetName": "amazon-ml-challenge-2026",
  "BudgetLimit": {"Amount": "${BUDGET_LIMIT}", "Unit": "USD"},
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}
EOF

cat > /tmp/notifications.json <<EOF
[
  {
    "Notification": {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": 75},
    "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "${ALERT_EMAIL}"}]
  },
  {
    "Notification": {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": 90},
    "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "${ALERT_EMAIL}"}]
  }
]
EOF

aws budgets create-budget \
  --account-id "$AWS_ACCOUNT_ID" \
  --budget file:///tmp/budget.json \
  --notifications-with-subscribers file:///tmp/notifications.json

echo "Budget alarm created: alerts at 75% (\$$(($BUDGET_LIMIT * 75 / 100))) and 90% (\$$(($BUDGET_LIMIT * 90 / 100))) of \$${BUDGET_LIMIT}."
