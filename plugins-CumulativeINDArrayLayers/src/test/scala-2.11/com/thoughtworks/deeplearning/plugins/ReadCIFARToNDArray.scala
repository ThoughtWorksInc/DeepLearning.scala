package com.thoughtworks.deeplearning.plugins

import java.io._
import java.net.URL

import sys.process._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.rauschig.jarchivelib.{Archiver, ArchiverFactory}

import scala.collection.immutable
import scala.collection.immutable.IndexedSeq

object ReadCIFARToNDArray {

  val currentPath
  : String = new java.io.File(".").getCanonicalPath + "/src/main/resources/"

  /**
    * 原始文件字节
    */
  lazy val originalCIFAR10FileBytesArray: Array[Array[Byte]] = {
    for (fileIndex <- 1 to 5) yield {

      val fileName = currentPath + "/cifar-10-batches-bin" + "/data_batch_" + fileIndex + ".bin"

      if (!new File(fileName).exists()) {
        downloadDataAndUnzipIfNotExist(
          "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
          currentPath,
          "cifar-10-binary.tar.gz"
        )
      }
      val inputStream = new FileInputStream(fileName)
      readFromInputStream(inputStream, 10000, 3073)
    }
  }.toArray

  lazy val originalCIFAR100FileBytesArray: Array[Byte] = {

    val fileName = currentPath + "/cifar-100-binary" + "/train.bin"

    ifCIFAR100NotExistThenDownload(fileName)
    val inputStream = new FileInputStream(fileName)
    readFromInputStream(inputStream, 50000, 3074)
  }

  /**
    * 中心化过后的图片
    */
  lazy val pixelBytesCIFAR10Array: Array[Array[Array[Double]]] = {
    for (fileIndex <- 0 until 5) yield {
      val originalFileBytes = originalCIFAR10FileBytesArray(fileIndex)
      for (index <- 0 until 10000) yield {
        val beginIndex = index * 3073 + 1
        normalizePixel(originalFileBytes.slice(beginIndex, beginIndex + 3072))
      }
    }.toArray
  }.toArray

  lazy val pixelBytesCIFAR100Array: Array[Array[Double]] = {
    for (index <- 0 until 50000) yield {
      val beginIndex = index * 3074 + 2
      normalizePixel(
        originalCIFAR100FileBytesArray.slice(beginIndex, beginIndex + 3072))
    }
  }.toArray

  /**
    * 图片对应的label
    */
  lazy val labelBytesCIFAR10Array: Array[Array[Int]] = {
    for (fileIndex <- 0 until 5) yield {
      val originalFileBytes = originalCIFAR10FileBytesArray(fileIndex)
      for (index <- 0 until 10000) yield {
        val beginIndex = index * 3073
        originalFileBytes(beginIndex).toInt
      }
    }.toArray
  }.toArray

  lazy val coarseLabelBytesCIFAR100Array: Array[Int] = {
    for (index <- 0 until 50000) yield {
      val beginIndex = index * 3074
      originalCIFAR100FileBytesArray(beginIndex).toInt
    }
  }.toArray

  lazy val fineLabelBytesCIFAR100Array: Array[Int] = {
    for (index <- 0 until 50000) yield {
      val beginIndex = index * 3074 + 1
      originalCIFAR100FileBytesArray(beginIndex).toInt
    }
  }.toArray

  val random = new util.Random

  /**
    * 从inputStream中读取byte
    *
    * @param inputStream
    * @return
    */
  def readFromInputStream(inputStream: InputStream,
                          batchCount: Int,
                          labelAndLabelSize: Int): Array[Byte] = {
    try {
      val bytes = Array.range(0, labelAndLabelSize * batchCount).map(_.toByte)
      inputStream.read(bytes)
      bytes
    } finally {
      inputStream.close()
    }
  }

  /**
    * 数据集不存在的话下载并解压
    *
    * @param path
    * @param url
    * @param fileName
    */
  //noinspection ScalaUnusedSymbol
  def downloadDataAndUnzipIfNotExist(url: String,
                                     path: String,
                                     fileName: String): Unit = {
    println("downloading from " + url)
    val result = new URL(url) #> new File(path + fileName) !!

    println("unzip" + fileName + "...")
    val archiver: Archiver = ArchiverFactory.createArchiver("tar", "gz")
    archiver.extract(new File(path + fileName), new File(path))
    println("download and unzip done.")
  }

  /**
    * 从CIFAR10文件中读图片和其对应的标签
    *
    * @param fileName CIFAR10文件名
    * @param count    要读取多少个图片和其标签
    * @return input :: expectedOutput :: HNil
    */
  def readFromCIFAR10Resource(fileName: String,
                              count: Int): (INDArray, INDArray) = {

    val filePathName = currentPath + fileName

    if (!new File(filePathName).exists()) {
      downloadDataAndUnzipIfNotExist(
        "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
        currentPath,
        "cifar-10-binary.tar.gz"
      )
    }

    val inputStream = new FileInputStream(filePathName)
    try {
      val bytes = Array.range(0, 3073 * count).map(_.toByte)
      inputStream.read(bytes)

      val labels: Seq[Double] =
        for (index <- 0 until count) yield {
          bytes(index * 3073).toDouble
        }

      val pixels: Seq[Seq[Double]] =
        for (index <- 0 until count) yield {
          for (item <- 1 until 3073) yield {
            normalizePixel(bytes(index * 3073 + item).toDouble)
          }
        }

      val labelsArray = labels.toNDArray.reshape(count, 1)
      val pixelsArray = pixels.toNDArray

      (pixelsArray, labelsArray)
    } finally {
      inputStream.close()
    }
  }

  /**
    * 从CIFAR100文件中读图片和其对应的标签
    *
    * @param fileName CIFAR100文件名
    * @param count    要读取多少个图片和其标签
    * @return (INDArray, INDArray, INDArray)
    */
  def readFromCIFAR100Resource(fileName: String,
                               count: Int): (INDArray, INDArray, INDArray) = {

    val filePath = currentPath + fileName

    ifCIFAR100NotExistThenDownload(filePath)

    val inputStream = new FileInputStream(filePath)
    try {
      val bytes = Array.range(0, 3074 * count).map(_.toByte)
      inputStream.read(bytes)

      val coarseLabels =
        for (index <- 0 until count) yield {
          bytes(index * 3074).toDouble
        }

      val fineLabels =
        for (index <- 0 until count) yield {
          bytes(index * 3074 + 1).toDouble
        }

      val pixels: Seq[Seq[Double]] =
        for (index <- 0 until count) yield {
          for (item <- 2 until 3074) yield {
            normalizePixel(bytes(index * 3074 + item).toDouble)
          }
        }

      val coarseLabelsArray = coarseLabels.toNDArray.reshape(count, 1)
      val fineLabelsArray = fineLabels.toNDArray.reshape(count, 1)
      val pixelsArray = pixels.toNDArray

      (pixelsArray, coarseLabelsArray, fineLabelsArray)
    } finally {
      inputStream.close()
    }
  }

  def initCIFAR100KeyMap(
                          labels: immutable.IndexedSeq[(Int, Int)]): Map[Int, Map[Int, Int]] = {
    labels.distinct
      .groupBy(_._1)
      .mapValues(_.map(_._2))
      .mapValues(_.zipWithIndex.toMap)
  }

  def getAllLabelsFromTestData: immutable.IndexedSeq[(Int, Int)] = {
    val filePath = currentPath + "/cifar-100-binary/test.bin"
    ifCIFAR100NotExistThenDownload(filePath)
    val inputStream = new FileInputStream(filePath)
    val bytes: Array[Byte] = Array.range(0, 3074 * 1000).map(_.toByte)
    try {
      inputStream.read(bytes)
      for (index <- 0 until 1000) yield {
        (bytes(index * 3074).toInt, bytes(index * 3074 + 1).toInt)
      }
    } finally {
      inputStream.close()
    }
  }

  lazy val coarseAndFineLabels: Map[Int, Map[Int, Int]] = {
    initCIFAR100KeyMap(getAllLabelsFromTestData)
  }

  case class ImageAndFineLabelINDArray(imageData: INDArray, fineLabel: INDArray)

  case class TestImageAndLabels(imageData: INDArray, coarse: Int, fine: Int)

  def readTestDataFromCIFAR100(
                                fileName: String,
                                count: Int): immutable.IndexedSeq[TestImageAndLabels] = {

    val filePath = currentPath + fileName

    ifCIFAR100NotExistThenDownload(filePath)
    val inputStream = new FileInputStream(filePath)
    try {
      val bytes = Array.range(0, 3074 * count).map(_.toByte)
      inputStream.read(bytes)
      for (index <- 0 until count) yield {
        val coarseLabel = bytes(index * 3074).toInt
        val fineLabel = bytes(index * 3074 + 1).toInt
        TestImageAndLabels(
          (for (item <- 2 until 3074) yield {
            normalizePixel(bytes(index * 3074 + item).toDouble)
          }).toNDArray
            .reshape(1, 3, 32, 32),
          coarseLabel,
          coarseAndFineLabels(coarseLabel)(fineLabel)
        )
      }
    } finally {
      inputStream.close()
    }
  }

  case class TrainData(image: INDArray,
                       coarseLabel: INDArray,
                       fineLabel: INDArray,
                       coarseClass: Int)

  def processBatchData(originalLabelAndData: Array[(Int, Int, Array[Double])])
  : Map[Int, ImageAndFineLabelINDArray] = {
    val grouped: Map[Int, immutable.IndexedSeq[(Int, Int, Array[Double])]] =
      originalLabelAndData.toIndexedSeq.groupBy(_._1)

    for (data <- grouped) yield {
      val (coarseLabel, originData) = data
      val size = originData.size

      val fineLabel: IndexedSeq[Int] =
        for (index <- 0 until size)
          yield {
            val origin = originData(index)
            coarseAndFineLabels(origin._1)(origin._2)
          }

      val originINDArray: IndexedSeq[IndexedSeq[Double]] =
        for (index <- 0 until size) yield {
          originData(index)._3.toIndexedSeq
        }

      coarseLabel ->
        ImageAndFineLabelINDArray(
          originINDArray.toNDArray
            .reshape(size, 3, 32, 32),
          Utils.makeVectorized(fineLabel.toNDArray.reshape(size, 1), 5))
    }
  }

  def readTrainData(
                     slicedIndexArray: Array[Int]): Array[(Int, Int, Array[Double])] = {
    for (index <- slicedIndexArray) yield {
      (coarseLabelBytesCIFAR100Array(index),
        fineLabelBytesCIFAR100Array(index),
        pixelBytesCIFAR100Array(index))
    }
  }

  private def processBatchDataToTrainData(
                                           data: Map[Int, ImageAndFineLabelINDArray]) = {
    (for (item <- data) yield {
      val (coarseLabel, ImageAndFineLabelINDArray(imageINDArray, fineLabel)) =
        item
      TrainData(imageINDArray,
        ReadCIFARToNDArray.getVectorizedLabel(coarseLabel, 20),
        fineLabel,
        coarseLabel)
    }).toIndexedSeq
  }

  def processSGDTrainData(
                           slicedIndexArray: Array[Int]): immutable.IndexedSeq[TrainData] = {
    val batchData: Array[(Int, Int, Array[Double])] =
      readTrainData(slicedIndexArray)
    processBatchDataToTrainData(processBatchData(batchData))
  }

  def getVectorizedLabel(data: Int, numberOfClasses: Int): INDArray = {
    Utils.makeVectorized(Array(data).toNDArray, numberOfClasses)
  }

  private def ifCIFAR100NotExistThenDownload(filePath: String): Unit = {
    if (!new File(filePath).exists()) {
      downloadDataAndUnzipIfNotExist(
        "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz",
        currentPath,
        "cifar-100-binary.tar.gz"
      )
    }
  }

  /**
    * 归一化pixel数据
    *
    * @param pixel
    * @return
    */
  def normalizePixel(pixel: Double): Double = {
    (if (pixel < 0) {
      pixel + 256
    } else {
      pixel
    }) / 256
  }

  /**
    * 归一化数组的pixel数据
    *
    * @param original
    * @return
    */
  def normalizePixel(original: Array[Byte]): Array[Double] = {
    for (pixel <- original) yield {
      normalizePixel(pixel)
    }
  }

  /**
    * 随机获取count个train数据
    *
    * @return
    */
  def getSGDCIFAR10TrainNDArray(
                                 randomIndexArray: Array[Int]): (INDArray, INDArray) = {
    //生成0到4的随机数
    val randomIndex = random.nextInt(5)
    val labelBytes = labelBytesCIFAR10Array(randomIndex)
    val pixelBytes = pixelBytesCIFAR10Array(randomIndex)
    val count = randomIndexArray.length

    val labels: Seq[Int] =
      for (index <- 0 until count) yield {
        labelBytes(randomIndexArray(index))
      }

    val pixels: Seq[Seq[Double]] =
      for (index <- 0 until count) yield {
        pixelBytes(randomIndexArray(index)).toList
      }

    val labelsNDArray = labels.toNDArray.reshape(count, 1)
    val pixelsNDArray = pixels.toNDArray

    (pixelsNDArray, labelsNDArray)
  }

  def getSGDCIFAR100TrainNDArray(
                                  randomIndexArray: Array[Int]): (INDArray, INDArray, INDArray) = {
    val count = randomIndexArray.length

    val coarseLabels: Seq[Int] =
      for (index <- 0 until count) yield {
        coarseLabelBytesCIFAR100Array(randomIndexArray(index))
      }

    val fineLabels: Seq[Int] =
      for (index <- 0 until count) yield {
        fineLabelBytesCIFAR100Array(randomIndexArray(index))
      }

    val pixels: Seq[Seq[Double]] =
      for (index <- 0 until count) yield {
        pixelBytesCIFAR100Array(randomIndexArray(index)).toList
      }

    val coarseLabelsNDArray =
      coarseLabels.toNDArray.reshape(count, 1)
    val fineLabelsNDArray =
      fineLabels.toNDArray.reshape(count, 1)
    val pixelsNDArray = pixels.toNDArray

    (pixelsNDArray, coarseLabelsNDArray, fineLabelsNDArray)
  }
}
