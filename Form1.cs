using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Arayuz_h2o
{
    public partial class Comparing_images : Form
    {
        public Comparing_images()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void exitbutton_Click(object sender, EventArgs e)
        {
            Application.Exit();

        }

        private void textBox2_TextChanged(object sender, EventArgs e)
        {

        }

        private void button2_Click(object sender, EventArgs e)
        {

            string img2018Path = textBox2.Text;
            string img2021Path = textBox1.Text;

            // Python script file path
            string pythonScriptPath = "main.py";

            string outputImagePath = "/output.png";

            // Python script arguments
            string arguments = $"{pythonScriptPath} \"{img2018Path}\" \"{img2021Path}\" \"{outputImagePath}\"";

            // Python betiğini başlatma
            Process process = new Process();
            process.StartInfo.FileName = "python";// Python yürütücüsünün adı
            process.StartInfo.Arguments = arguments;
            process.StartInfo.UseShellExecute = false;
            process.StartInfo.RedirectStandardOutput = true;
            process.StartInfo.CreateNoWindow = true;

            // İşlemi başlat
            process.Start();

            // İşlemi bekletme
            process.WaitForExit();

            //// İşlem çıkışını okuma (isteğe bağlı)
            //string output = process.StandardOutput.ReadToEnd();
            //MessageBox.Show(output); // Python betiğinin çıktısını göstermek için MessageBox kullanabilirsiniz
            if (File.Exists(outputImagePath))
            {
                pictureBox1.Image = Image.FromFile(outputImagePath);
            }
            else
            {
                MessageBox.Show("Image not found.");
            }
            // İşlem kaynaklarını temizleme
            process.Close();
        }
    }
}
