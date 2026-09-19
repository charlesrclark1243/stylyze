import { HttpErrorResponse } from '@angular/common/http';
import { Component, inject, OnDestroy, signal } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatSliderModule } from '@angular/material/slider';
import { MatSnackBar } from '@angular/material/snack-bar';
import { MatToolbarModule } from '@angular/material/toolbar';
import { firstValueFrom } from 'rxjs';
import { CompareSlider } from './components/compare-slider/compare-slider';
import { ImagePicker } from './components/image-picker/image-picker';
import { STYLE_PRESETS } from './presets';
import { Stylyze } from './services/stylyze';

interface Result {
  before: string;
  after: string;
}

@Component({
  imports: [
    CompareSlider,
    ImagePicker,
    MatButtonModule,
    MatIconModule,
    MatProgressBarModule,
    MatSliderModule,
    MatToolbarModule,
  ],
  selector: 'app-root',
  styleUrl: './app.scss',
  templateUrl: './app.html',
})
export class App implements OnDestroy {
  private readonly stylyzeService = inject(Stylyze);
  private readonly snackBar = inject(MatSnackBar);

  protected readonly stylePresets = STYLE_PRESETS;

  protected readonly content = signal<File | null>(null);
  protected readonly style = signal<File | null>(null);
  protected readonly result = signal<Result | null>(null);
  protected readonly loading = signal(false);

  // Style strength: 1 applies the full style, lower values keep more of the photo. Below 1 is gentler on faces
  protected readonly alpha = signal(0.8);

  protected formatStrength(value: number) {
    return `${Math.round(value * 100)}%`;
  }

  protected async stylyze() {
    const content = this.content();
    const style = this.style();
    if (!content || !style) return;

    this.loading.set(true);
    try {
      const blob = await firstValueFrom(this.stylyzeService.stylyze(content, style, this.alpha()));
      this.revokeResult();
      // Keeps its own URL for the content image, so the comparison still matches after a new one is picked
      this.result.set({ before: URL.createObjectURL(content), after: URL.createObjectURL(blob) });
    } catch (err) {
      this.snackBar.open(await errorMessage(err), 'Dismiss', { duration: 6000 });
    } finally {
      this.loading.set(false);
    }
  }

  ngOnDestroy() {
    this.revokeResult();
  }

  private revokeResult() {
    const result = this.result();
    if (!result) return;
    URL.revokeObjectURL(result.before);
    URL.revokeObjectURL(result.after);
  }
}

// With responseType 'blob', error bodies arrive as a Blob too, so FastAPI's JSON detail
// has to be read out of it. nginx and Caddy send HTML instead, so this returns null for them.
async function errorDetail(err: HttpErrorResponse): Promise<string | null> {
  if (!(err.error instanceof Blob)) return null;

  try {
    const body = JSON.parse(await err.error.text());
    return typeof body.detail === 'string' ? body.detail : null;
  } catch {
    return null;
  }
}

async function errorMessage(err: unknown): Promise<string> {
  const fallback = 'Something went wrong. Please try again.';
  if (!(err instanceof HttpErrorResponse)) return fallback;

  // The backend's own message is the most specific one available, when there is one
  const detail = await errorDetail(err);

  switch (err.status) {
    case 429: {
      // Two sources: nginx sheds bursts per IP, and the backend refuses work when its
      // single inference slot is busy. Only the latter sends Retry-After.
      const retryAfter = Number(err.headers.get('Retry-After'));
      const wait =
        Number.isFinite(retryAfter) && retryAfter > 0 ? `${retryAfter} seconds` : 'a moment';
      return (
        detail ?? `Busy right now - only one image is stylized at a time. Try again in ${wait}.`
      );
    }
    case 413:
      return detail ?? 'Those images are too large. Each one must be under 10 MB.';
    case 415:
      return detail ?? 'Only JPEG and PNG images are supported.';
    case 503:
      return detail ?? 'The stylizer is still starting up. Try again in a minute.';
    case 504:
      return 'That took too long to stylize. Try again, or use a smaller image.';
    case 0:
      // Angular reports network failures and CORS errors as status 0
      return 'Could not reach the server. Check your connection and try again.';
    default:
      return detail ?? fallback;
  }
}
