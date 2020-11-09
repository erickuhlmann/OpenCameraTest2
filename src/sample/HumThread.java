package sample;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.SourceDataLine;

/**
 * Emits a humming sound while the thread is active.
 */
public class HumThread extends Thread
{
    private static final int SAMPLE_RATE = 16 * 1024;

    private volatile boolean soundStarted = false;

    @Override
    public void run()
    {
        while (true)
        {
            if (soundStarted)
            {
                try
                {
                    playSound();
                }
                catch (LineUnavailableException e)
                {
                    ;
                }
            }
        }
    }

    public void startSound()
    {
        soundStarted = true;
    }

    public void stopSound()
    {
        soundStarted = false;
    }

    private void playSound() throws LineUnavailableException
    {
        final AudioFormat af = new AudioFormat(SAMPLE_RATE, 8, 1, true, true);
        SourceDataLine line = AudioSystem.getSourceDataLine(af);
        line.open(af, SAMPLE_RATE);
        line.start();

        double freq = 150;

        for (int i = 0; i < 50; i++)
        {
            byte[] toneBuffer = createSinWaveBuffer(freq);
            int count = line.write(toneBuffer, 0, toneBuffer.length);
        }

        line.drain();
        line.close();

    }

    public static byte[] createSinWaveBuffer(double freq)
    {
        double waveLen = 1.0/freq;
        int samples = (int) Math.round(waveLen * 5 * SAMPLE_RATE);
        byte[] output = new byte[samples];
        double period = SAMPLE_RATE / freq;
        for (int i = 0; i < output.length; i++)
        {
            double angle = 2.0 * Math.PI * i / period;
            output[i] = (byte)(Math.sin(angle) * 127f);
        }
        return output;
    }

}
