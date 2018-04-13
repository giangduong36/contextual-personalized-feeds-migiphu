# Generated by Django 2.0.2 on 2018-04-13 02:18

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('recsys', '0003_auto_20180413_0119'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='comment',
            name='from_id',
        ),
        migrations.AddField(
            model_name='comment',
            name='page_id',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='recsys.Page'),
        ),
    ]
