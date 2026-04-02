import { test, expect } from '@playwright/test';

test('homepage has correct title and loads properly', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/MiroFish/);
  await expect(page.locator('#app')).toBeVisible();
});
